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
  * 5-fold CV; the best checkpoint per fold is saved, and the 5 are
    ensembled for final predictions (see predict.py).

Two things this file is careful about, both of which were wrong before:

  * **Normalisation is fitted per fold, on the training rows only.** The
    imputer is fitted inside `train_one_fold` and stored in the checkpoint,
    so evaluation and prediction reuse the exact constants the model was
    trained under.
  * **The 20% decay is applied to the schedule's base LR**, not to
    `param_groups[...]["lr"]`, which a cosine scheduler overwrites on its
    next step. See `WarmRestartSchedule`.

Out-of-fold validation predictions are written to
`outputs/oof_predictions.csv`. They are what evaluate.py reports metrics on
and what predict.py calibrates against — in-sample predictions would
flatter both.

Usage:
    python -m src.train --epochs 150 --folds 5 --full-length   # the paper
    python -m src.train --epochs 2 --folds 2 --max-rows 400     # smoke test
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Sampler

from .dataset import WINDOW_SIZE, FeatureTracks, build_dataset, collate_batch
from .features import ScoreImputer
from .model import DeepATM

PROCESSED_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("outputs")

CONSEQUENCE_MIX = {"Missense": 0.90, "Synonymous": 0.05, "Nonsense": 0.05}

# STAR★Methods hyperparameters.
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
T_0 = 10          # initial restart cycle length, epochs
T_MULT = 2        # cycle length doubles after each restart
RESTART_DECAY = 0.8  # "the learning rate was reduced by 20% after each restart"


class WarmRestartSchedule:
    """Cosine annealing with warm restarts, where each restart also decays
    the peak learning rate by 20%.

    Written out rather than wrapped around `CosineAnnealingWarmRestarts`
    because the composition is the part the paper specifies and the part the
    previous implementation silently lost: multiplying `param_groups["lr"]`
    by 0.8 has no lasting effect, since the scheduler recomputes lr from
    `base_lrs` on its next step. Here the decay applies to the base.

    With T_0=10, T_mult=2 the restarts land at epochs 10, 30, 70, 150, so a
    150-epoch run sees four cycles with peaks 1e-3, 8e-4, 6.4e-4, 5.12e-4.
    """

    def __init__(self, base_lr: float = LEARNING_RATE, t_0: int = T_0,
                 t_mult: int = T_MULT, decay: float = RESTART_DECAY):
        if t_0 < 1:
            raise ValueError("t_0 must be >= 1")
        self.base_lr, self.t_0, self.t_mult, self.decay = base_lr, t_0, t_mult, decay

    def cycle_at(self, epoch: int) -> tuple[int, int, int]:
        """Return (cycle index, epoch within cycle, cycle length)."""
        cycle, start, length = 0, 0, self.t_0
        while epoch >= start + length:
            start += length
            length *= self.t_mult
            cycle += 1
        return cycle, epoch - start, length

    def lr_at(self, epoch: int) -> float:
        cycle, t_cur, t_i = self.cycle_at(epoch)
        peak = self.base_lr * (self.decay ** cycle)
        return peak * 0.5 * (1.0 + math.cos(math.pi * t_cur / t_i))

    def apply(self, optimizer: torch.optim.Optimizer, epoch: int) -> float:
        lr = self.lr_at(epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr


class ConsequenceBalancedSampler(Sampler[list[int]]):
    """Yields batch indices drawn ~90/5/5 missense/synonymous/nonsense, per
    STAR★Methods: "The training data were dynamically sampled in each batch
    ... to consist of 90% missense variants, 5% synonymous variants, and 5%
    nonsense variants."

    At batch_size=20 that is 18/1/1. Classes absent from the frame (which
    happens under --max-rows) are dropped and their weight redistributed,
    so a smoke run still trains rather than crashing.
    """

    def __init__(self, consequences: pd.Series, batch_size: int, n_batches: int, seed: int = 0):
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.rng = np.random.default_rng(seed)
        values = consequences.to_numpy()
        by_class = {cls: np.where(values == cls)[0] for cls in CONSEQUENCE_MIX}
        self.by_class = {k: v for k, v in by_class.items() if len(v) > 0}
        if not self.by_class:
            raise ValueError("no rows with a recognised Variant_consequence")
        total = sum(CONSEQUENCE_MIX[k] for k in self.by_class)
        self.weights = {k: CONSEQUENCE_MIX[k] / total for k in self.by_class}

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            batch: list[int] = []
            for cls, frac in self.weights.items():
                n = max(1, round(self.batch_size * frac))
                pool = self.by_class[cls]
                batch.extend(int(i) for i in self.rng.choice(pool, size=n, replace=len(pool) < n))
            yield batch[: self.batch_size]


def run_fingerprint(**kwargs) -> dict:
    """The run-defining parameters a resume file must agree with.

    Resuming into a checkpoint trained under different settings would silently
    produce a model that matches neither configuration, so a mismatch is a
    hard error rather than a fresh start.
    """
    return {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
            for k, v in sorted(kwargs.items())}


def save_resume_state(path: Path, state: dict) -> None:
    """Atomically write mid-fold training state.

    Written every epoch, so a spot reclamation costs one epoch rather than the
    fold. The temp-then-replace matters: the instance can die *during* the
    write, and a truncated resume file would take the fold with it.

    Everything stored is a tensor, a primitive, or a container of those, so
    the file loads back under `weights_only=True`.
    """
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_resume_state(path: Path, fingerprint: dict) -> dict | None:
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("fingerprint") != fingerprint:
        raise RuntimeError(
            f"{path} was written under different settings and cannot be resumed.\n"
            f"  saved: {state.get('fingerprint')}\n"
            f"  now:   {fingerprint}\n"
            "Delete the resume files to start over, or re-run with the original flags."
        )
    return state


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run_epoch(model, loader, device, optimizer=None, scaler=None):
    """One pass. Returns (mean loss, predictions, targets)."""
    is_train = optimizer is not None
    model.train(is_train)
    use_amp = scaler is not None and scaler.is_enabled()
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
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(aa_seq, domain_seq, coords, mut_position, aux_scores)
                loss = torch.nn.functional.mse_loss(pred.float(), target)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
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
        preds_all.append(pred.detach().float().cpu().numpy())
        targets_all.append(target.detach().cpu().numpy())

    preds = np.concatenate(preds_all) if preds_all else np.empty(0, dtype=np.float32)
    targets = np.concatenate(targets_all) if targets_all else np.empty(0, dtype=np.float32)
    return total_loss / max(n, 1), preds, targets


def train_one_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tracks: FeatureTracks,
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    patience: int,
    fold_idx: int,
    window_size: int | None,
    seed: int,
    use_coordinates: bool = True,
    num_workers: int = 0,
    resume_path: Path | None = None,
    fingerprint: dict | None = None,
) -> dict:
    torch.manual_seed(seed + fold_idx)

    # Fitted on this fold's training rows only. Handing it the whole frame
    # first, as the previous version did, leaks validation statistics into
    # the normalisation constants. Refitting on resume is safe because the fit
    # is deterministic in the fold's training rows and draws no randomness.
    imputer = ScoreImputer().fit(train_df)

    train_ds = build_dataset(train_df, tracks, imputer, window_size=window_size)
    val_ds = build_dataset(val_df, tracks, imputer, window_size=window_size)

    n_batches = max(1, len(train_ds) // batch_size)
    sampler = ConsequenceBalancedSampler(
        train_df["Variant_consequence"], batch_size, n_batches, seed=seed + fold_idx
    )
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_ds, batch_sampler=sampler, collate_fn=collate_batch, **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, **loader_kwargs
    )

    model = DeepATM(n_aux_scores=imputer.n_features, use_coordinates=use_coordinates).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    schedule = WarmRestartSchedule()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    best_state = None
    best_preds = None
    best_epoch = -1
    since_improvement = 0
    lr_history: list[float] = []
    start_epoch = 0

    state = load_resume_state(resume_path, fingerprint) if resume_path else None
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng_state"])
        sampler.rng.bit_generator.state = state["sampler_rng_state"]
        start_epoch = int(state["next_epoch"])
        best_val_loss = float(state["best_val_loss"])
        best_epoch = int(state["best_epoch"])
        best_state = state["best_model"]
        best_preds = state["best_preds"].numpy()
        since_improvement = int(state["since_improvement"])
        lr_history = [float(x) for x in state["lr_history"]]
        print(f"[fold {fold_idx}] resuming at epoch {start_epoch} "
              f"(best val_loss {best_val_loss:.4f} at epoch {best_epoch})")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        lr = schedule.apply(optimizer, epoch)
        lr_history.append(lr)

        train_loss, _, _ = run_epoch(model, train_loader, device, optimizer, scaler)
        val_loss, val_preds, val_targets = run_epoch(model, val_loader, device)

        r = pearson_corr(val_preds, val_targets)
        elapsed = time.time() - t0
        remaining = (epochs - epoch - 1) * elapsed
        print(
            f"[fold {fold_idx}] epoch {epoch:3d}  lr={lr:.2e}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_pearson={r:.3f}  ({elapsed:.1f}s, <={remaining / 60:.0f}m left in fold)",
            flush=True,
        )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            best_preds = val_preds
            since_improvement = 0
        else:
            since_improvement += 1

        if resume_path is not None:
            save_resume_state(resume_path, {
                "fingerprint": fingerprint,
                "fold": fold_idx,
                "next_epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "sampler_rng_state": sampler.rng.bit_generator.state,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "best_model": best_state,
                "best_preds": torch.from_numpy(np.asarray(best_preds)),
                "since_improvement": since_improvement,
                "lr_history": lr_history,
                "done": False,
            })

        if not improved and since_improvement >= patience:
            print(f"[fold {fold_idx}] early stopping at epoch {epoch}")
            break

    if best_state is None:  # epochs == 0, or resumed past the end
        raise RuntimeError("no epoch completed; --epochs must be >= 1")
    model.load_state_dict(best_state)

    return {
        "model": model,
        "imputer": imputer,
        "val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "val_preds": best_preds,
        "lr_history": lr_history,
    }


def save_checkpoint(path: Path, fold: dict, window_size: int | None, use_coordinates: bool) -> None:
    """Persist weights *and* the normalisation constants they were trained
    under. A checkpoint without its imputer is not reusable — evaluation
    would have to refit, which is how the leak got in originally."""
    imputer: ScoreImputer = fold["imputer"]
    torch.save(
        {
            "state_dict": fold["model"].state_dict(),
            # Stored as plain lists, not numpy arrays, so the checkpoint can
            # be read back with weights_only=True (no arbitrary unpickling).
            "imputer": {
                "with_flags": imputer.with_flags,
                "means": imputer.means,
                "mu": imputer.mu.tolist(),
                "sigma": imputer.sigma.tolist(),
            },
            "config": {
                "n_aux_scores": imputer.n_features,
                "window_size": window_size,
                "use_coordinates": use_coordinates,
            },
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[DeepATM, ScoreImputer, dict]:
    blob = torch.load(path, map_location=device, weights_only=True)
    config = blob["config"]
    model = DeepATM(
        n_aux_scores=config["n_aux_scores"],
        use_coordinates=config.get("use_coordinates", True),
    ).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    saved = blob["imputer"]
    imputer = ScoreImputer(
        with_flags=saved["with_flags"],
        means=saved["means"],
        mu=np.asarray(saved["mu"], dtype=np.float64),
        sigma=np.asarray(saved["sigma"], dtype=np.float64),
    )
    return model, imputer, config


def load_training_frame(path: Path, max_rows: int | None, seed: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `python -m src.data_prep` first.")
    df = pd.read_csv(path)
    df = df.dropna(subset=["position", "alt_aa", "function_score"])
    df["position"] = df["position"].astype(int)
    if max_rows:
        df = df.sample(n=min(max_rows, len(df)), random_state=seed)
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max-rows", type=int, default=None, help="Subsample for a quick smoke test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-csv", type=Path, default=PROCESSED_DIR / "train.csv")
    parser.add_argument(
        "--window", type=int, default=WINDOW_SIZE,
        help=f"Residues of context around the mutation (default {WINDOW_SIZE}). "
             "A windowed run is not the paper's model.",
    )
    parser.add_argument(
        "--full-length", action="store_true",
        help="Use all 3,056 residues, as the paper does. Needs a GPU.",
    )
    parser.add_argument(
        "--no-coordinates", action="store_true",
        help="Ablation (M6): drop the structural branch.",
    )
    parser.add_argument("--tag", default="", help="Suffix for checkpoint filenames, e.g. 'nocoord'")
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers. 0 on CPU; 4-8 on a GPU, where building the "
             "3,056-residue tracks in Python is otherwise the bottleneck.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Checkpoint every epoch and continue from the last one. Required "
             "for spot instances, where the run can be reclaimed mid-fold.",
    )
    parser.add_argument(
        "--allow-cpu-full-length", action="store_true",
        help="Permit --full-length without a GPU. It will take days.",
    )
    args = parser.parse_args()

    if args.folds < 2:
        raise SystemExit(
            f"--folds must be >= 2 (got {args.folds}); with one fold there are no "
            "remaining rows to train on."
        )

    window_size = None if args.full_length else args.window
    use_coordinates = not args.no_coordinates
    suffix = f"_{args.tag}" if args.tag else ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  window: {window_size or 'full length (3056)'}  "
          f"coordinates: {use_coordinates}")
    if device.type == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
    if window_size is not None:
        print("  NOTE: windowed run - results are not directly comparable to the paper (D8).")
    if window_size is None and device.type != "cuda" and not args.allow_cpu_full_length:
        raise SystemExit(
            "--full-length on CPU is ~100x slower than the windowed path and will not "
            "finish in any useful time. Run it on a CUDA device, or pass "
            "--allow-cpu-full-length if you really mean to."
        )

    run_started = time.time()
    df = load_training_frame(args.train_csv, args.max_rows, args.seed)
    print(f"training rows: {len(df):,}")
    print(df["Variant_consequence"].value_counts().to_string())

    tracks = FeatureTracks.load()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(df))
    folds = np.array_split(indices, args.folds)

    oof_pred = np.full(len(df), np.nan, dtype=np.float32)
    oof_fold = np.full(len(df), -1, dtype=np.int64)
    summary = []

    for fold_idx in range(args.folds):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[i] for i in range(args.folds) if i != fold_idx])
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        fingerprint = run_fingerprint(
            epochs=args.epochs, folds=args.folds, batch_size=args.batch_size,
            patience=args.patience, window_size=window_size, seed=args.seed,
            use_coordinates=use_coordinates, fold=fold_idx,
            n_train=len(train_df), n_val=len(val_df),
        )
        resume_path = CHECKPOINT_DIR / f"resume_fold{fold_idx}{suffix}.pt" if args.resume else None

        # A fold that finished in an earlier attempt is not retrained. Its
        # out-of-fold predictions come back from the resume file, so the OOF
        # frame is identical to an uninterrupted run's.
        done_state = load_resume_state(resume_path, fingerprint) if resume_path else None
        if done_state is not None and done_state.get("done"):
            oof_pred[val_idx] = done_state["best_preds"].numpy()
            oof_fold[val_idx] = fold_idx
            summary.append({
                "fold": fold_idx,
                "val_loss": float(done_state["best_val_loss"]),
                "best_epoch": int(done_state["best_epoch"]),
                "n_train": len(train_df),
                "n_val": len(val_df),
                "lr_history": [float(x) for x in done_state["lr_history"]],
                "resumed": "already complete",
            })
            print(f"[fold {fold_idx}] already complete "
                  f"(val_loss={float(done_state['best_val_loss']):.4f}) - skipping")
            continue

        result = train_one_fold(
            train_df, val_df, tracks, device,
            epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
            fold_idx=fold_idx, window_size=window_size, seed=args.seed,
            use_coordinates=use_coordinates, num_workers=args.num_workers,
            resume_path=resume_path, fingerprint=fingerprint,
        )

        oof_pred[val_idx] = result["val_preds"]
        oof_fold[val_idx] = fold_idx
        save_checkpoint(
            CHECKPOINT_DIR / f"deepatm_fold{fold_idx}{suffix}.pt",
            result, window_size, use_coordinates,
        )
        if resume_path is not None:
            state = torch.load(resume_path, map_location="cpu", weights_only=True)
            state["done"] = True
            save_resume_state(resume_path, state)

        summary.append({
            "fold": fold_idx,
            "val_loss": result["val_loss"],
            "best_epoch": result["best_epoch"],
            "n_train": len(train_df),
            "n_val": len(val_df),
            "lr_history": result["lr_history"],
        })
        print(f"[fold {fold_idx}] best val_loss={result['val_loss']:.4f} "
              f"at epoch {result['best_epoch']}")

    # Out-of-fold predictions: every row scored by the one fold that did not
    # train on it. evaluate.py and predict.py both read this rather than
    # re-predicting in-sample.
    # hg38_pos/Ref/Alt identify the SNV uniquely; Protein_change does not —
    # several synonymous SNVs share one protein change, which would fan out
    # any join keyed on it (see src.compare_runs).
    oof = df[["hg38_pos", "Ref", "Alt", "Protein_change", "position", "ref_aa",
              "alt_aa", "Variant_consequence", "function_score"]].copy()
    oof["fold"] = oof_fold
    oof["oof_prediction"] = oof_pred
    oof_path = OUTPUT_DIR / f"oof_predictions{suffix}.csv"
    oof.to_csv(oof_path, index=False)

    with open(OUTPUT_DIR / f"train_summary{suffix}.json", "w") as fh:
        json.dump({
            "folds": summary,
            "window_size": window_size,
            "use_coordinates": use_coordinates,
            "seed": args.seed,
            "n_rows": len(df),
            # Recorded so a result can be traced back to the run that made it.
            # `window_size: null` plus `n_rows: 21715` is the only combination
            # that is comparable to the paper.
            "epochs": args.epochs,
            "folds_requested": args.folds,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "device": device.type,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "torch_version": torch.__version__,
            "wall_clock_seconds": round(time.time() - run_started, 1),
        }, fh, indent=2)

    print(f"\nmean val loss across {args.folds} fold(s): "
          f"{np.mean([s['val_loss'] for s in summary]):.4f}")
    print(f"wrote {oof_path}")


if __name__ == "__main__":
    main()
