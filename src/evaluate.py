"""
Evaluate trained DeepATM checkpoints, per STAR★Methods "Performance
evaluation":

  * Pearson / Spearman between predicted and measured function scores, on
    held-out folds only, with the median across folds as the headline.
  * auROC separating ClinVar pathogenic/likely-pathogenic from
    benign/likely-benign on the 116-variant >=1-star test set, with 1,000
    bootstrap resamples for the confidence interval.
  * The 5 fold checkpoints ensembled (mean prediction) for the test-set
    numbers.
  * The same auROC for AlphaMissense and the other precomputed scores.

Four things this file is careful about, all of which were wrong before and
each of which alone invalidated the output:

  1. **Correlations are out-of-fold.** They are read from
     `outputs/oof_predictions.csv`, where each row was scored by the one fold
     that did not train on it. The previous version loaded `measured.csv` —
     the training data — and reported in-sample correlation.
  2. **The auROC sign.** A low function score means loss of ATM function,
     i.e. pathogenic. Scoring `preds` directly with pathogenic as the
     positive class yields 1 - AUC. Predictions are negated.
  3. **ClinVar labels are matched exactly.** "Conflicting classifications of
     pathogenicity" contains the substring "pathogenic" and was being counted
     as pathogenic; `data_prep.clinvar_label` matches against an explicit set.
  4. **Baselines are also model inputs.** AlphaMissense, EVE, REVEL and CADD
     are among the 16 scores fed to the head, so this comparison measures
     "does the transformer add anything on top of its own features", not
     independent superiority. Reported, with that caveat, as the paper does.

Usage:
    python -m src.evaluate
    python -m src.evaluate --checkpoints "checkpoints/deepatm_fold*_nocoord.pt" --tag nocoord
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .dataset import FeatureTracks, build_dataset, collate_batch
from .features import arcsinh_transform
from .model import DeepATM
from .train import load_checkpoint, pearson_corr

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs")

# The paper's headline numbers, for the side-by-side in the printed report.
PAPER = {
    "cv_pearson": 0.61,
    "test_auroc_1star": 0.95,
    "eda_correlation": 0.70,
}

# Precomputed scores usable as baselines, and the sign that makes "larger =
# more pathogenic". DeepATM predicts a function score, where *smaller* is more
# damaging, so its own predictions are negated the same way.
BASELINE_SIGN = {
    "AlphaMissense": +1,   # pathogenicity probability
    "REVEL": +1,           # pathogenicity score
    "CADD.phred": +1,      # deleteriousness
    "EVE_scores_ASM": +1,  # pathogenicity score
    "boostDM_score": +1,   # driver probability
}

# The paper compares against these four. Only AlphaMissense ships in Table S1;
# the rest need the dbNSFP slice (EXECUTION_PLAN.md 2.1).
PAPER_BASELINES = ["AlphaMissense", "ESM1b", "phyloP100", "PROVEAN"]


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------

@torch.no_grad()
def predict_with_model(
    model: DeepATM, imputer, df: pd.DataFrame, tracks: FeatureTracks,
    device: torch.device, window_size: int | None, batch_size: int = 32,
) -> np.ndarray:
    """Score `df` with one checkpoint, using that checkpoint's own imputer."""
    ds = build_dataset(df, tracks, imputer, window_size=window_size, has_target=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    out = []
    for batch in loader:
        pred = model(
            batch.aa_seq.to(device),
            batch.domain_seq.to(device),
            batch.coords.to(device),
            batch.mut_position.to(device),
            batch.aux_scores.to(device),
        )
        out.append(pred.float().cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


def load_ensemble(pattern: str, device: torch.device) -> tuple[list, int | None]:
    """Load every matching checkpoint with its fitted imputer.

    Each fold carries its own normalisation constants, so they cannot be
    shared — a single imputer across folds is exactly the leak M2 removed.
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No checkpoints matched {pattern} - run `python -m src.train` first."
        )
    members, windows = [], set()
    for path in paths:
        model, imputer, config = load_checkpoint(Path(path), device)
        members.append((Path(path).name, model, imputer))
        windows.add(config["window_size"])
    if len(windows) > 1:
        raise ValueError(f"checkpoints disagree on window size: {windows}")
    return members, windows.pop()


def predict_ensemble(members, df, tracks, device, window_size) -> np.ndarray:
    per_model = [
        predict_with_model(model, imputer, df, tracks, device, window_size)
        for _, model, imputer in members
    ]
    return np.mean(per_model, axis=0)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def bootstrap_auroc(labels: np.ndarray, scores: np.ndarray, n: int = 1000, seed: int = 0):
    """Point estimate plus a percentile bootstrap CI.

    Resamples that end up single-class have no defined auROC and are skipped
    rather than counted as 0.5, which would drag the interval toward chance.
    """
    labels, scores = np.asarray(labels), np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return {"auroc": None, "ci_low": None, "ci_high": None, "n": int(len(labels))}

    point = float(roc_auc_score(labels, scores))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[idx])) < 2:
            continue
        draws.append(roc_auc_score(labels[idx], scores[idx]))
    return {
        "auroc": point,
        "ci_low": float(np.percentile(draws, 2.5)) if draws else None,
        "ci_high": float(np.percentile(draws, 97.5)) if draws else None,
        "n": int(len(labels)),
        "n_positive": int(labels.sum()),
        "n_resamples_used": len(draws),
    }


def cross_validation_metrics(oof_path: Path) -> dict:
    """Per-fold correlations on held-out predictions, plus the median.

    The paper reports 5-fold CV Pearson r ~0.61. Median across folds is what
    it plots; the pooled value is reported alongside because a median over
    five noisy folds is itself noisy.
    """
    if not oof_path.exists():
        raise FileNotFoundError(
            f"{oof_path} not found - run `python -m src.train` first. "
            "Correlations must come from out-of-fold predictions."
        )
    oof = pd.read_csv(oof_path).dropna(subset=["oof_prediction", "function_score"])
    y = arcsinh_transform(oof["function_score"].to_numpy(dtype=np.float64))
    p = oof["oof_prediction"].to_numpy(dtype=np.float64)

    per_fold = []
    for fold, part in oof.groupby("fold"):
        yy = arcsinh_transform(part["function_score"].to_numpy(dtype=np.float64))
        pp = part["oof_prediction"].to_numpy(dtype=np.float64)
        per_fold.append({
            "fold": int(fold),
            "n": int(len(part)),
            "pearson": pearson_corr(pp, yy),
            "spearman": float(spearmanr(pp, yy).correlation),
        })

    by_consequence = {}
    for cons, part in oof.groupby("Variant_consequence"):
        if len(part) < 10:
            continue
        yy = arcsinh_transform(part["function_score"].to_numpy(dtype=np.float64))
        pp = part["oof_prediction"].to_numpy(dtype=np.float64)
        by_consequence[str(cons)] = {"n": int(len(part)), "pearson": pearson_corr(pp, yy)}

    return {
        "n": int(len(oof)),
        "per_fold": per_fold,
        "median_pearson": float(np.median([f["pearson"] for f in per_fold])),
        "median_spearman": float(np.median([f["spearman"] for f in per_fold])),
        "pooled_pearson": pearson_corr(p, y),
        "pooled_spearman": float(spearmanr(p, y).correlation),
        "by_consequence": by_consequence,
    }


def clinvar_metrics(test_df: pd.DataFrame, preds: np.ndarray, seed: int = 0) -> dict:
    """auROC on the ClinVar test set, for DeepATM and each baseline score."""
    labels = test_df["clinvar_label"].to_numpy(dtype=int)

    # Negated: DeepATM predicts a function score, and a LOW function score is
    # the pathogenic direction. Passing preds unnegated gives 1 - AUC.
    results = {"DeepATM": bootstrap_auroc(labels, -preds, seed=seed)}

    baselines = {}
    for col, sign in BASELINE_SIGN.items():
        if col not in test_df.columns:
            continue
        values = pd.to_numeric(test_df[col], errors="coerce")
        keep = values.notna().to_numpy()
        if keep.sum() < 10 or len(np.unique(labels[keep])) < 2:
            baselines[col] = {"auroc": None, "note": "too few labelled non-missing rows"}
            continue
        entry = bootstrap_auroc(labels[keep], sign * values.to_numpy()[keep], seed=seed)
        entry["coverage"] = float(keep.mean())
        baselines[col] = entry

    for col in PAPER_BASELINES:
        if col not in baselines:
            baselines[col] = {"auroc": None, "note": "not in Table S1; needs the dbNSFP slice"}

    return {"deepatm": results["DeepATM"], "baselines": baselines}


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _fmt(value, spec=".3f", missing="  n/a"):
    return format(value, spec) if isinstance(value, (int, float)) else missing


def print_report(metrics: dict) -> None:
    cv = metrics["cross_validation"]
    print("\nCross-validation (out-of-fold, arcsinh-transformed function score)")
    for fold in cv["per_fold"]:
        print(f"  fold {fold['fold']}  n={fold['n']:>6,}  "
              f"r={_fmt(fold['pearson'])}  rho={_fmt(fold['spearman'])}")
    print(f"  median r   = {_fmt(cv['median_pearson'])}   paper {PAPER['cv_pearson']:.2f}")
    print(f"  median rho = {_fmt(cv['median_spearman'])}")
    print(f"  pooled r   = {_fmt(cv['pooled_pearson'])}  (n={cv['n']:,})")
    if cv["by_consequence"]:
        print("  by consequence: " + "  ".join(
            f"{k} r={_fmt(v['pearson'])} (n={v['n']:,})" for k, v in cv["by_consequence"].items()
        ))

    clin = metrics.get("clinvar_1star")
    if clin:
        d = clin["deepatm"]
        print(f"\nClinVar >=1-star test set (n={d['n']}, {d.get('n_positive')} P/LP)")
        print(f"  DeepATM auROC = {_fmt(d['auroc'])} "
              f"[{_fmt(d['ci_low'])}, {_fmt(d['ci_high'])}]   "
              f"paper {PAPER['test_auroc_1star']:.2f}")
        print("  baselines (note: these are also model inputs)")
        for name, entry in clin["baselines"].items():
            if entry.get("auroc") is None:
                print(f"    {name:<16} n/a   ({entry.get('note', 'unavailable')})")
            else:
                print(f"    {name:<16} {_fmt(entry['auroc'])} "
                      f"[{_fmt(entry['ci_low'])}, {_fmt(entry['ci_high'])}]  "
                      f"coverage {entry['coverage']:.0%}")

    print("\nNot yet reported: the >=2-star subset (n=68 in the paper) needs the "
          "ClinVar variant_summary join - EXECUTION_PLAN.md 2.3.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", default="checkpoints/deepatm_fold*.pt")
    parser.add_argument("--oof", type=Path, default=None,
                        help="Out-of-fold predictions (default: outputs/oof_predictions<tag>.csv)")
    parser.add_argument("--test-csv", type=Path, default=PROCESSED_DIR / "test.csv")
    parser.add_argument("--tag", default="", help="Suffix matching the training run, e.g. 'nocoord'")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-test-set", action="store_true",
                        help="Cross-validation metrics only; no checkpoint loading")
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    oof_path = args.oof or OUTPUT_DIR / f"oof_predictions{suffix}.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics: dict = {"cross_validation": cross_validation_metrics(oof_path)}

    if not args.skip_test_set:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        members, window_size = load_ensemble(args.checkpoints, device)
        print(f"Loaded {len(members)} checkpoint(s): {[n for n, _, _ in members]}")
        if window_size is not None:
            print(f"  window {window_size} - not the paper's full-length model (D8)")

        test_df = pd.read_csv(args.test_csv).dropna(subset=["position", "alt_aa", "clinvar_label"])
        test_df["position"] = test_df["position"].astype(int)
        tracks = FeatureTracks.load()
        preds = predict_ensemble(members, test_df, tracks, device, window_size)

        metrics["clinvar_1star"] = clinvar_metrics(test_df, preds, seed=args.seed)
        metrics["ensemble"] = {
            "n_checkpoints": len(members),
            "window_size": window_size,
            "checkpoints": [n for n, _, _ in members],
        }

        test_out = test_df[["Protein_change", "position", "ref_aa", "alt_aa",
                            "ClinVar_classification", "clinvar_label", "function_score"]].copy()
        test_out["ensemble_prediction"] = preds
        test_out.to_csv(OUTPUT_DIR / f"test_predictions{suffix}.csv", index=False)

    metrics["paper_targets"] = PAPER
    out_path = OUTPUT_DIR / f"metrics{suffix}.json"
    with open(out_path, "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)

    print_report(metrics)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
