"""
Train DeepATM with 5-fold cross-validation, per STAR★Methods "Model
training" and "Performance evaluation":

  * Optimizer: AdamW, initial lr 1e-3, weight decay 1e-2.
  * Schedule: cosine annealing with warm restarts, initial cycle length 10
    epochs; after each restart, lr *= 0.8 and cycle length doubles.
  * Up to 150 epochs; early stopping after 20 epochs without val-loss
    improvement.
  * Batches of 20, dynamically sampled per batch to be ~90% missense, 5%
    synonymous, 5% nonsense.
  * MSE loss (on the arcsinh-transformed function score).
  * Mixed precision + gradient clipping.
  * 5-fold CV; best checkpoint per fold saved; the 5 best models are
    ensembled (averaged) for final predictions (see predict.py).

Usage:
    python -m src.train --epochs 150 --folds 5
    python -m src.train --epochs 3 --folds 1 --max-rows 500   # quick smoke test
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Sampler

from .dataset import ATMVariantDataset, collate_batch, load_reference_sequence
from .features import ScoreImputer, build_coordinate_track, build_domain_track
from .model import DeepATM

PROCESSED_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")

CONSEQUENCE_MIX = {"Missense": 0.90, "Synonymous": 0.05, "Nonsense": 0.05}


class ConsequenceBalancedSampler(Sampler[int]):
    """Yields batch indices drawn ~90/5/5 missense/synonymous/nonsense, per
    STAR★Methods "The training data were dynamically sampled in each batch
    ... to consist of 90% missense variants, 5% synonymous variants, and 5%
    nonsense variants."
    """

    def __init__(self, consequences: pd.Series, batch_size: int, n_batches: int, seed: int = 0):
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.rng = np.random.default_rng(seed)
        self.by_class = {
            cls: np.where(consequences.to_numpy() == cls)[0]
            for cls in CONSEQUENCE_MIX
        }
        # Fall back to uniform sampling for any class with zero examples
        # (e.g. tiny smoke-test subsets).
        self.by_class = {k: v for k, v in self.by_class.items() if len(v) > 0}
        total_weight = sum(CONSEQUENCE_MIX[k] for k in self.by_class)
        self.weights = {k: CONSEQUENCE_MIX[k] / total_weight for k in self.by_class}

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for cls, frac in self.weights.items():
                n = max(1, round(self.batch_size * frac))
                pool = self.by_class[cls]
                batch.extend(self.rng.choice(pool, size=n, replace=len(pool) < n))
            batch = batch[: self.batch_size]
            yield batch


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run_epoch(model, loader, device, optimizer=None, scaler=None) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n = 0.0, 0
    preds_all, targets_all = [], []

    for batch in loader:
        aa_seq = batch.aa_seq.to(device)
        domain_seq = batch.domain_seq.to(device)
        coords = batch.coords.to(device)
        mut_position = batch.mut_position.to(device)
        aux_scores = batch.aux_scores.to(device)
        target = batch.target.to(device)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                pred = model(aa_seq, domain_seq, coords, mut_position, aux_scores)
                loss = torch.nn.functional.mse_loss(pred, target)

            if is_train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

        total_loss += loss.item() * aa_seq.size(0)
        n += aa_seq.size(0)
        preds_all.append(pred.detach().cpu().numpy())
        targets_all.append(target.detach().cpu().numpy())

    return total_loss / max(n, 1), np.concatenate(preds_all), np.concatenate(targets_all)


def train_one_fold(
    train_df, val_df, reference_seq, domain_track, coord_track, score_imputer,
    device, epochs, batch_size, patience, fold_idx,
):
    train_ds = ATMVariantDataset(train_df, reference_seq, domain_track, coord_track, score_imputer)
    val_ds = ATMVariantDataset(val_df, reference_seq, domain_track, coord_track, score_imputer)

    n_batches_per_epoch = max(1, len(train_ds) // batch_size)
    sampler = ConsequenceBalancedSampler(train_df["Variant_consequence"], batch_size, n_batches_per_epoch)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    model = DeepATM().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, _, _ = run_epoch(model, train_loader, device, optimizer, scaler if device.type == "cuda" else None)
        val_loss, val_preds, val_targets = run_epoch(model, val_loader, device)
        scheduler.step(epoch)

        # cosine-annealing-with-restarts: decay base LR by 20% after each restart
        if epoch > 0 and epoch % 10 == 0:
            for group in optimizer.param_groups:
                group["lr"] *= 0.8

        r = pearson_corr(val_preds, val_targets)
        dt = time.time() - t0
        print(f"[fold {fold_idx}] epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_pearson={r:.3f}  ({dt:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[fold {fold_idx}] early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, best_val_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max-rows", type=int, default=None, help="Subsample for a quick smoke test")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    measured_path = PROCESSED_DIR / "measured.csv"
    if not measured_path.exists():
        raise FileNotFoundError(f"{measured_path} not found — run `python -m src.data_prep` first.")

    df = pd.read_csv(measured_path)
    df = df.dropna(subset=["position", "alt_aa"])
    if args.max_rows:
        df = df.sample(n=min(args.max_rows, len(df)), random_state=args.seed).reset_index(drop=True)

    reference_seq = load_reference_sequence(df)
    domain_track = build_domain_track()
    coord_track = build_coordinate_track()
    score_imputer = ScoreImputer().fit(df)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(df))

    if args.folds == 1:
        # Simple 80/20 split -- useful for quick smoke tests; the paper's
        # reported metrics use --folds 5 (true k-fold CV).
        split = int(len(indices) * 0.8)
        fold_bounds = [indices[split:]]
        train_val_pairs = [(indices[:split], indices[split:])]
    else:
        fold_bounds = np.array_split(indices, args.folds)
        train_val_pairs = [
            (np.concatenate([fold_bounds[i] for i in range(args.folds) if i != fold_idx]), fold_bounds[fold_idx])
            for fold_idx in range(args.folds)
        ]

    fold_losses = []
    for fold_idx, (train_idx, val_idx) in enumerate(train_val_pairs):
        train_df, val_df = df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)

        model, val_loss = train_one_fold(
            train_df, val_df, reference_seq, domain_track, coord_track, score_imputer,
            device, args.epochs, args.batch_size, args.patience, fold_idx,
        )
        fold_losses.append(val_loss)
        torch.save(model.state_dict(), CHECKPOINT_DIR / f"deepatm_fold{fold_idx}.pt")

    print(f"mean val loss across {args.folds} fold(s): {np.mean(fold_losses):.4f}")


if __name__ == "__main__":
    main()
