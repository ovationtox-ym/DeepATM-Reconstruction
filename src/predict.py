"""
Generate eDA scores for the 4,421 unevaluated ATM SNVs, per STAR★Methods
"Predicting the effects of unevaluated ATM variants":

  1. Run the ensembled DeepATM model on the held-out (`predicted.csv`,
     i.e. DeepATM_predicted == "Yes") variants to get raw predictions.
  2. Rank-align those raw predictions against the measured function score
     distribution (a simple rank-based rescaling here; the paper uses
     generalized additive regression for this step -- see
     `rank_align_gam` for a smoothed alternative if you have `pygam`
     installed).
  3. Classify into functional / intermediate / non-functional using the
     same cutoffs as the measured function scores (-1.360, -0.912).

Usage:
    python -m src.predict --checkpoints checkpoints/deepatm_fold*.pt
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .dataset import ATMVariantDataset, load_reference_sequence
from .evaluate import load_ensemble, predict_dataset
from .features import ScoreImputer, build_coordinate_track, build_domain_track, inverse_arcsinh_transform

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs")

NON_FUNCTIONAL_CUTOFF = -1.360
INTERMEDIATE_CUTOFF = -0.912


def classify(score: float) -> str:
    if score < NON_FUNCTIONAL_CUTOFF:
        return "Non-functional"
    if score < INTERMEDIATE_CUTOFF:
        return "Intermediate"
    return "Functional"


def rank_align(raw_preds: np.ndarray, measured_preds: np.ndarray, measured_scores: np.ndarray) -> np.ndarray:
    """Map `raw_preds` onto the measured function-score scale by matching
    percentile rank against the model's own predictions on the measured set.
    This is a simplified stand-in for the paper's generalized-additive-model
    rank alignment; swap in `pygam.LinearGAM` for a closer match if desired.
    """
    order = np.argsort(measured_preds)
    sorted_model_preds = measured_preds[order]
    sorted_scores = measured_scores[order]
    ranks = np.searchsorted(sorted_model_preds, raw_preds).clip(0, len(sorted_scores) - 1)
    return sorted_scores[ranks]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=str, default="checkpoints/deepatm_fold*.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_paths = sorted(glob.glob(args.checkpoints))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints matched {args.checkpoints} — run `python -m src.train` first.")
    models = load_ensemble(checkpoint_paths, device)

    measured_df = pd.read_csv(PROCESSED_DIR / "measured.csv").dropna(subset=["position", "alt_aa"])
    predicted_df = pd.read_csv(PROCESSED_DIR / "predicted.csv").dropna(subset=["position", "alt_aa"])

    reference_seq = load_reference_sequence(measured_df)
    domain_track = build_domain_track()
    coord_track = build_coordinate_track()
    score_imputer = ScoreImputer().fit(measured_df)

    measured_ds = ATMVariantDataset(measured_df, reference_seq, domain_track, coord_track, score_imputer)
    measured_preds = predict_dataset(models, measured_ds, device)  # transformed-space, for rank alignment

    unevaluated_ds = ATMVariantDataset(
        predicted_df, reference_seq, domain_track, coord_track, score_imputer, has_target=False,
    )
    raw_preds = predict_dataset(models, unevaluated_ds, device)

    eda_scores = rank_align(raw_preds, measured_preds, measured_df["Combined_score"].to_numpy(dtype=np.float32))

    result = predicted_df.copy()
    result["eDA_score_reconstruction"] = eda_scores
    result["classification_reconstruction"] = [classify(s) for s in eda_scores]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "eda_scores_reconstruction.csv"
    result.to_csv(out_path, index=False)
    print(f"Wrote {len(result)} predictions to {out_path}")

    # If the paper's own DeepATM_predicted rows carry the published
    # Combined_score too, report agreement as a sanity check.
    if "Combined_score" in predicted_df.columns and predicted_df["Combined_score"].notna().any():
        published = predicted_df["Combined_score"].to_numpy(dtype=np.float32)
        r = np.corrcoef(eda_scores, published)[0, 1]
        print(f"Correlation with published eDA scores (reference only, not used in training): r = {r:.3f}")


if __name__ == "__main__":
    main()
