"""
Generate eDA scores for the 4,421 unevaluated ATM SNVs, per STAR★Methods
"Predicting the effects of unevaluated ATM variants":

  1. Ensemble-predict the unevaluated variants (`predict.csv`, the rows with
     DeepATM_predicted == "Yes").
  2. Rank-align raw predictions to the measured function-score distribution,
     then fit a generalized additive model to that relationship and map the
     unmeasured predictions through it.
  3. Classify at the published cutoffs: -1.360 (the 5th percentile of
     synonymous variants) and -0.912 (Youden's index for nonsense vs
     synonymous).

Two corrections to the earlier version:

  * **Calibration is fitted on out-of-fold predictions.** It previously used
    the ensemble's in-sample predictions on the measured set, so the
    calibration curve inherited the ensemble's memorisation of its own
    training data and was systematically too confident.
  * **A GAM replaces the `searchsorted` step function.** The paper specifies
    generalized additive regression; the nearest-rank lookup it had was a
    piecewise-constant map that can only emit function scores that were
    literally observed.

The headline validation is at the end: the paper's own eDA scores for these
same 4,421 variants are in Table S1 and were never used in training, so the
correlation printed there is a direct variant-level comparison against the
original model's output.

Usage:
    python -m src.predict
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from .dataset import FeatureTracks
from .evaluate import load_ensemble, predict_ensemble
from .features import arcsinh_transform
from .train import pearson_corr

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs")

# Verified against the PDF. Both are on the function-score scale.
NON_FUNCTIONAL_CUTOFF = -1.360  # 5th percentile of synonymous variants
INTERMEDIATE_CUTOFF = -0.912    # Youden's index, nonsense vs synonymous


def classify(scores: np.ndarray) -> np.ndarray:
    labels = np.full(len(scores), "Functional", dtype=object)
    labels[scores < INTERMEDIATE_CUTOFF] = "Intermediate"
    labels[scores < NON_FUNCTIONAL_CUTOFF] = "Non-functional"
    return labels


class RankAlignedGAM:
    """The paper's calibration: rank-align, then fit a GAM.

    `fit` takes out-of-fold model predictions and the measured function
    scores for the same variants. Rank alignment maps each prediction to the
    function score at its own percentile — a monotone quantile match that
    fixes scale and skew — and the GAM is then fitted from raw prediction to
    that rank-aligned target, giving a smooth invertible curve that can be
    applied to variants with no measurement.

    Falls back to isotonic regression if `pygam` is unavailable, which is the
    same monotone idea without the spline smoothness.
    """

    def __init__(self, n_splines: int = 20):
        self.n_splines = n_splines
        self.model = None
        self.backend = None

    @staticmethod
    def _rank_align(preds: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Quantile-match predictions onto the observed score distribution."""
        ranks = np.argsort(np.argsort(preds))
        quantiles = (ranks + 0.5) / len(preds)
        return np.quantile(np.sort(scores), quantiles)

    def fit(self, preds: np.ndarray, scores: np.ndarray) -> "RankAlignedGAM":
        preds = np.asarray(preds, dtype=np.float64)
        target = self._rank_align(preds, np.asarray(scores, dtype=np.float64))
        try:
            from pygam import LinearGAM, s

            # Monotone increasing: a higher predicted score must never map to
            # a lower function score, or the calibration would reorder
            # variants that the model itself ranked.
            self.model = LinearGAM(
                s(0, n_splines=self.n_splines, constraints="monotonic_inc")
            ).fit(preds.reshape(-1, 1), target)
            self.backend = "pygam.LinearGAM"
        except ImportError:
            from sklearn.isotonic import IsotonicRegression

            self.model = IsotonicRegression(out_of_bounds="clip").fit(preds, target)
            self.backend = "sklearn.IsotonicRegression (pygam not installed)"
        return self

    def transform(self, preds: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("RankAlignedGAM.transform called before fit()")
        preds = np.asarray(preds, dtype=np.float64)
        if self.backend.startswith("pygam"):
            return self.model.predict(preds.reshape(-1, 1))
        return self.model.predict(preds)


def load_oof(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python -m src.train` first. Calibration "
            "must be fitted on out-of-fold predictions, not in-sample ones."
        )
    oof = pd.read_csv(path).dropna(subset=["oof_prediction", "function_score"])
    if oof.empty:
        raise ValueError(f"{path} has no usable rows")
    return oof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", default="checkpoints/deepatm_fold*.pt")
    parser.add_argument("--predict-csv", type=Path, default=PROCESSED_DIR / "predict.csv")
    parser.add_argument("--oof", type=Path, default=None)
    parser.add_argument("--tag", default="", help="Suffix matching the training run")
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    oof_path = args.oof or OUTPUT_DIR / f"oof_predictions{suffix}.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members, window_size = load_ensemble(args.checkpoints, device)
    print(f"Loaded {len(members)} checkpoint(s)")
    if window_size is not None:
        print(f"  window {window_size} - not the paper's full-length model (D8)")

    # Calibrate on held-out predictions.
    oof = load_oof(oof_path)
    calibrator = RankAlignedGAM().fit(
        oof["oof_prediction"].to_numpy(), oof["function_score"].to_numpy()
    )
    print(f"Calibration fitted on {len(oof):,} out-of-fold predictions "
          f"({calibrator.backend})")

    # Sanity check: how well does the calibration recover the measured scores
    # it was fitted on? Out-of-fold, so this is not circular.
    calibrated_oof = calibrator.transform(oof["oof_prediction"].to_numpy())
    measured = oof["function_score"].to_numpy()
    print(f"  calibrated vs measured (out-of-fold): "
          f"r={pearson_corr(calibrated_oof, measured):.3f}  "
          f"rho={spearmanr(calibrated_oof, measured).correlation:.3f}")

    # Score the unevaluated variants.
    predict_df = pd.read_csv(args.predict_csv).dropna(subset=["position", "alt_aa"])
    predict_df["position"] = predict_df["position"].astype(int)
    tracks = FeatureTracks.load()
    raw = predict_ensemble(members, predict_df, tracks, device, window_size)
    eda = calibrator.transform(raw)

    result = predict_df.copy()
    result["raw_prediction"] = raw
    result["eDA_score_reconstruction"] = eda
    result["classification_reconstruction"] = classify(eda)

    out_path = OUTPUT_DIR / f"eda_scores_reconstruction{suffix}.csv"
    result.to_csv(out_path, index=False)
    print(f"\nWrote {len(result):,} predictions to {out_path}")

    counts = result["classification_reconstruction"].value_counts()
    print("  " + "  ".join(f"{k}: {v:,}" for k, v in counts.items()))

    summary = {
        "n_predicted": int(len(result)),
        "calibration_backend": calibrator.backend,
        "n_calibration_rows": int(len(oof)),
        "oof_calibrated_pearson": pearson_corr(calibrated_oof, measured),
        "classification_counts": {str(k): int(v) for k, v in counts.items()},
        "window_size": window_size,
    }

    # The single best validation this project has: the paper's own eDA scores
    # for these same variants, which this reconstruction never saw.
    published = pd.to_numeric(predict_df["published_eda_score"], errors="coerce").to_numpy()
    keep = ~np.isnan(published)
    if keep.sum() > 10:
        r = pearson_corr(eda[keep], published[keep])
        rho = float(spearmanr(eda[keep], published[keep]).correlation)
        # Also in the transformed space the model actually works in, since a
        # calibration curve can flatter or flatten the raw agreement.
        r_raw = pearson_corr(raw[keep], arcsinh_transform(published[keep]))
        print(f"\nAgainst the paper's published eDA scores (n={int(keep.sum()):,}, "
              "never used in training):")
        print(f"  calibrated eDA   r={r:.3f}  rho={rho:.3f}")
        print(f"  raw (arcsinh)    r={r_raw:.3f}")
        summary["vs_published_eda"] = {
            "n": int(keep.sum()), "pearson": r, "spearman": rho, "pearson_raw": r_raw,
        }

    with open(OUTPUT_DIR / f"predict_summary{suffix}.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
