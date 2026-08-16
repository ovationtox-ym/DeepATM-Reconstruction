"""
Evaluate trained DeepATM checkpoints, per STAR★Methods "Performance
evaluation":

  * Pearson / Spearman correlation between predicted and measured function
    scores on held-out folds.
  * auROC classifying ClinVar pathogenic/likely-pathogenic vs.
    benign/likely-benign variants, for >=1-star and >=2-star ClinVar
    subsets.
  * The 5 fold checkpoints are ensembled (averaged prediction) for the
    final reported numbers.

Usage:
    python -m src.evaluate --checkpoints checkpoints/deepatm_fold*.pt
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .dataset import ATMVariantDataset, collate_batch, load_reference_sequence
from .features import ScoreImputer, build_coordinate_track, build_domain_track, inverse_arcsinh_transform
from .model import DeepATM
from .train import pearson_corr

PROCESSED_DIR = Path("data/processed")


@torch.no_grad()
def predict_dataset(models: list[DeepATM], ds: ATMVariantDataset, device: torch.device, batch_size: int = 32) -> np.ndarray:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    all_preds = []
    for batch in loader:
        aa_seq = batch.aa_seq.to(device)
        domain_seq = batch.domain_seq.to(device)
        coords = batch.coords.to(device)
        mut_position = batch.mut_position.to(device)
        aux_scores = batch.aux_scores.to(device)

        ensemble_preds = torch.stack([
            m(aa_seq, domain_seq, coords, mut_position, aux_scores) for m in models
        ])
        all_preds.append(ensemble_preds.mean(dim=0).cpu().numpy())
    return np.concatenate(all_preds)


def load_ensemble(checkpoint_paths: list[str], device: torch.device) -> list[DeepATM]:
    models = []
    for path in checkpoint_paths:
        model = DeepATM().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=str, default="checkpoints/deepatm_fold*.pt")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_paths = sorted(glob.glob(args.checkpoints))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints matched {args.checkpoints} — run `python -m src.train` first.")
    models = load_ensemble(checkpoint_paths, device)
    print(f"Loaded {len(models)} checkpoint(s): {checkpoint_paths}")

    df = pd.read_csv(PROCESSED_DIR / "measured.csv").dropna(subset=["position", "alt_aa"])
    if args.max_rows:
        df = df.sample(n=min(args.max_rows, len(df)), random_state=0).reset_index(drop=True)

    reference_seq = load_reference_sequence(df)
    domain_track = build_domain_track()
    coord_track = build_coordinate_track()
    score_imputer = ScoreImputer().fit(df)

    ds = ATMVariantDataset(df, reference_seq, domain_track, coord_track, score_imputer)
    preds_transformed = predict_dataset(models, ds, device)
    preds = inverse_arcsinh_transform(preds_transformed)
    targets = df["Combined_score"].to_numpy(dtype=np.float32)

    r_pearson = pearson_corr(preds, targets)
    r_spearman = spearmanr(preds, targets).correlation
    print(f"Pearson r  = {r_pearson:.3f}")
    print(f"Spearman r = {r_spearman:.3f}")

    clinvar = df["ClinVar_classification"].fillna("")
    is_plp = clinvar.str.contains("athogenic", case=False, na=False) & ~clinvar.str.contains("enign", case=False, na=False)
    is_blb = clinvar.str.contains("enign", case=False, na=False) & ~clinvar.str.contains("athogenic", case=False, na=False)
    mask = is_plp | is_blb
    if mask.sum() >= 10 and is_plp[mask].nunique() > 1:
        auc = roc_auc_score(is_plp[mask].astype(int), preds[mask])
        print(f"auROC (P/LP vs B/LB, n={mask.sum()}) = {auc:.3f}")
    else:
        print("Not enough ClinVar-labeled variants in this subset for auROC.")


if __name__ == "__main__":
    main()
