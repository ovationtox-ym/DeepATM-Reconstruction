"""
Random-forest baseline (EXECUTION_PLAN.md M6).

The paper reports that a random forest trained on the precomputed scores for
missense variants alone reaches Pearson r ~0.55, against DeepATM's ~0.61.
That gap is one of the two cheapest ways to falsify a bad reconstruction: a
transformer that matches 0.61 while a forest on the same features also
matches 0.61 has learned nothing the scores did not already contain.

To make the comparison exact, this uses **the same fold assignment as
`src.train`** — same seed, same permutation, same `array_split` over the full
training frame — and then restricts to missense. So the per-fold numbers
printed here line up one-to-one with DeepATM's missense-only per-fold
numbers in `outputs/metrics.json`.

The imputer is fitted per fold on training rows only, exactly as in training.

Usage:
    python -m src.baseline_rf --seed 0 --folds 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

from .features import AUX_SCORE_COLS, ScoreImputer, arcsinh_transform
from .train import load_training_frame, pearson_corr

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs")

PAPER_RF_PEARSON = 0.55


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=PROCESSED_DIR / "train.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=500)
    args = parser.parse_args()

    df = load_training_frame(args.train_csv, args.max_rows, args.seed)

    # Identical fold construction to src.train.main, so the folds match.
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(df))
    folds = np.array_split(indices, args.folds)

    is_missense = (df["Variant_consequence"] == "Missense").to_numpy()
    y = arcsinh_transform(df["function_score"].to_numpy(dtype=np.float64))
    print(f"training rows: {len(df):,}  of which missense: {int(is_missense.sum()):,}")

    oof = np.full(len(df), np.nan)
    oof_fold = np.full(len(df), -1, dtype=np.int64)
    per_fold = []

    for fold_idx in range(args.folds):
        val_idx = folds[fold_idx][is_missense[folds[fold_idx]]]
        train_idx = np.concatenate([folds[i] for i in range(args.folds) if i != fold_idx])
        train_idx = train_idx[is_missense[train_idx]]
        if len(val_idx) < 2 or len(train_idx) < 10:
            print(f"[rf fold {fold_idx}] too few missense rows; skipped")
            continue

        imputer = ScoreImputer().fit(df.iloc[train_idx])
        x_train = imputer.transform(df.iloc[train_idx])
        x_val = imputer.transform(df.iloc[val_idx])

        forest = RandomForestRegressor(
            n_estimators=args.n_estimators,
            random_state=args.seed + fold_idx,
            n_jobs=-1,
        ).fit(x_train, y[train_idx])

        pred = forest.predict(x_val)
        oof[val_idx] = pred
        oof_fold[val_idx] = fold_idx

        r = pearson_corr(pred, y[val_idx])
        rho = float(spearmanr(pred, y[val_idx]).correlation)
        per_fold.append({"fold": fold_idx, "n": int(len(val_idx)), "pearson": r, "spearman": rho})
        print(f"[rf fold {fold_idx}] n={len(val_idx):>6,}  r={r:.3f}  rho={rho:.3f}")

    if not per_fold:
        raise RuntimeError("no fold produced a usable missense split")

    scored = ~np.isnan(oof)
    summary = {
        "model": "RandomForestRegressor",
        "n_estimators": args.n_estimators,
        "features": AUX_SCORE_COLS,
        "n_features_after_flags": int(ScoreImputer().n_features),
        "subset": "missense only",
        "n": int(scored.sum()),
        "per_fold": per_fold,
        "median_pearson": float(np.median([f["pearson"] for f in per_fold])),
        "median_spearman": float(np.median([f["spearman"] for f in per_fold])),
        "pooled_pearson": pearson_corr(oof[scored], y[scored]),
        "paper_pearson": PAPER_RF_PEARSON,
        "seed": args.seed,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = df.loc[scored, ["hg38_pos", "Ref", "Alt", "Protein_change", "position",
                          "ref_aa", "alt_aa", "Variant_consequence", "function_score"]].copy()
    out["fold"] = oof_fold[scored]
    out["oof_prediction"] = oof[scored]
    out.to_csv(OUTPUT_DIR / "oof_predictions_rf.csv", index=False)

    with open(OUTPUT_DIR / "baseline_rf.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print(f"\nmedian Pearson r = {summary['median_pearson']:.3f}   "
          f"paper {PAPER_RF_PEARSON:.2f}")
    print(f"pooled Pearson r = {summary['pooled_pearson']:.3f}  (n={summary['n']:,})")
    print(f"wrote {OUTPUT_DIR / 'baseline_rf.json'}")


if __name__ == "__main__":
    main()
