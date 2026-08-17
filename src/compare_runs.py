"""
Compare two runs' out-of-fold predictions (EXECUTION_PLAN.md M6).

The paper's structural ablation — a transformer without Ca coordinates scores
lower, p = 0.032 — is the sharpest single check on this reconstruction. A
model that reaches r ~0.61 but shows no gap when the coordinate branch is
removed has probably learned something other than what DeepATM learned.

Both runs must share a fold assignment (same --seed and --folds in
`src.train`) so the comparison is paired variant by variant.

Two tests are reported, because they answer different questions:

  * **Paired bootstrap over variants** — resample variants with replacement,
    recompute both correlations on the same resample, and count how often the
    difference reverses sign. This is the sensitive test; n is in the
    thousands, so it has power.
  * **Wilcoxon signed-rank over folds** — the fold-level test. With 5 folds
    its smallest attainable p-value is 0.0625, so it can never reach 0.05;
    it is reported for completeness, not as the verdict. (The paper's
    p = 0.032 is therefore not a 5-fold Wilcoxon either.)

Usage:
    python -m src.train --tag full                    # with coordinates
    python -m src.train --no-coordinates --tag nocoord
    python -m src.compare_runs outputs/oof_predictions_full.csv \\
                               outputs/oof_predictions_nocoord.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .features import arcsinh_transform
from .train import pearson_corr

OUTPUT_DIR = Path("outputs")

# Genomic identity. NOT Protein_change — several synonymous SNVs share one
# protein change, so a join keyed on it fans out into a partial cross product
# and silently compares different variants against each other.
KEY = ["hg38_pos", "Ref", "Alt"]


def load_and_align(path_a: Path, path_b: Path) -> pd.DataFrame:
    """Inner-join two out-of-fold files on variant identity.

    Joining rather than assuming row order is deliberate: the two runs may
    have dropped different rows, and a positional zip would silently compare
    different variants.
    """
    a = pd.read_csv(path_a).dropna(subset=["oof_prediction", "function_score"])
    b = pd.read_csv(path_b).dropna(subset=["oof_prediction", "function_score"])
    for name, frame in (("A", a), ("B", b)):
        if frame.duplicated(KEY).any():
            raise ValueError(f"run {name} has duplicate {KEY} rows; the join would fan out")
    merged = a.merge(b, on=KEY, suffixes=("_a", "_b"), validate="one_to_one")
    if merged.empty:
        raise ValueError("the two runs share no variants")
    if not np.allclose(merged["function_score_a"], merged["function_score_b"]):
        raise ValueError("function scores disagree between runs - different data?")
    if (merged["fold_a"] != merged["fold_b"]).any():
        raise ValueError(
            "fold assignments differ between runs; rerun both with the same "
            "--seed and --folds so the comparison is paired"
        )
    return merged


def paired_bootstrap(y, pred_a, pred_b, n: int = 1000, seed: int = 0) -> dict:
    observed = pearson_corr(pred_a, y) - pearson_corr(pred_b, y)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        d = pearson_corr(pred_a[idx], y[idx]) - pearson_corr(pred_b[idx], y[idx])
        if not np.isnan(d):
            diffs.append(d)
    diffs = np.asarray(diffs)
    # Two-sided: how often does the resampled difference cross zero?
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "delta_pearson": float(observed),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_value": float(min(p, 1.0)),
        "n_resamples": int(len(diffs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path, help="Out-of-fold CSV for run A (e.g. with coordinates)")
    parser.add_argument("run_b", type=Path, help="Out-of-fold CSV for run B (e.g. ablated)")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--subset", choices=["all", "Missense", "Synonymous", "Nonsense"], default="all")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    merged = load_and_align(args.run_a, args.run_b)
    if args.subset != "all":
        merged = merged[merged["Variant_consequence_a"] == args.subset]
        if len(merged) < 10:
            raise ValueError(f"only {len(merged)} rows in subset {args.subset}")

    y = arcsinh_transform(merged["function_score_a"].to_numpy(dtype=np.float64))
    pred_a = merged["oof_prediction_a"].to_numpy(dtype=np.float64)
    pred_b = merged["oof_prediction_b"].to_numpy(dtype=np.float64)

    per_fold = []
    for fold, part in merged.groupby("fold_a"):
        yy = arcsinh_transform(part["function_score_a"].to_numpy(dtype=np.float64))
        per_fold.append({
            "fold": int(fold),
            "n": int(len(part)),
            "pearson_a": pearson_corr(part["oof_prediction_a"].to_numpy(dtype=np.float64), yy),
            "pearson_b": pearson_corr(part["oof_prediction_b"].to_numpy(dtype=np.float64), yy),
        })

    boot = paired_bootstrap(y, pred_a, pred_b, n=args.bootstrap, seed=args.seed)

    fold_diffs = [f["pearson_a"] - f["pearson_b"] for f in per_fold]
    if len(fold_diffs) >= 3 and any(d != 0 for d in fold_diffs):
        stat, p_wilcoxon = wilcoxon(fold_diffs)
        wilcoxon_result = {"statistic": float(stat), "p_value": float(p_wilcoxon),
                           "n_folds": len(fold_diffs)}
    else:
        wilcoxon_result = {"p_value": None, "note": "too few folds"}

    print(f"{args.label_a}: {args.run_a}")
    print(f"{args.label_b}: {args.run_b}")
    print(f"subset: {args.subset}   paired variants: {len(merged):,}\n")
    print(f"  {'fold':<6}{'n':>8}  {args.label_a:>8}  {args.label_b:>8}  {'delta':>8}")
    for f in per_fold:
        print(f"  {f['fold']:<6}{f['n']:>8,}  {f['pearson_a']:>8.3f}  "
              f"{f['pearson_b']:>8.3f}  {f['pearson_a'] - f['pearson_b']:>8.3f}")

    print(f"\n  median r  {args.label_a} = {np.median([f['pearson_a'] for f in per_fold]):.3f}")
    print(f"  median r  {args.label_b} = {np.median([f['pearson_b'] for f in per_fold]):.3f}")
    print(f"\n  pooled delta r = {boot['delta_pearson']:+.4f} "
          f"[{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]  "
          f"p = {boot['p_value']:.4f}  (paired bootstrap, {boot['n_resamples']} resamples)")
    if wilcoxon_result.get("p_value") is not None:
        print(f"  fold-level Wilcoxon p = {wilcoxon_result['p_value']:.4f} "
              f"(floor 0.0625 at 5 folds - cannot reach 0.05)")

    summary = {
        "run_a": str(args.run_a), "run_b": str(args.run_b),
        "label_a": args.label_a, "label_b": args.label_b,
        "subset": args.subset, "n_paired": int(len(merged)),
        "per_fold": per_fold,
        "paired_bootstrap": boot,
        "fold_wilcoxon": wilcoxon_result,
        "paper_ablation_p": 0.032,
    }
    out_path = args.out or OUTPUT_DIR / "ablation_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
