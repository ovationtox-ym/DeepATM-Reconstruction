"""
PyTorch Dataset that turns rows of the processed `measured.csv` /
`predicted.csv` tables into model-ready tensors: a windowed slice of the
mutant amino-acid sequence, the matching domain/coordinate tracks, the
mutated position within that window, and the auxiliary score vector.

STAR★Methods trains DeepATM over the full 3,056-residue sequence; here the
window is configurable (`WINDOW_SIZE`) so the same code can run either in
full-length mode (set WINDOW_SIZE = None, needs a GPU) or in a
laptop/CPU-friendly cropped mode (default).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import (
    N_RESIDUES,
    ScoreImputer,
    arcsinh_transform,
    build_coordinate_track,
    build_domain_track,
    encode_amino_acid,
)

WINDOW_SIZE = 255  # odd number -> symmetric window around the mutated residue; None = full length


def load_reference_sequence(measured_df: pd.DataFrame, n_residues: int = N_RESIDUES) -> list[str]:
    """Reconstruct the wild-type amino acid at each position from `ref_aa`
    values seen across all rows (every position should have >=1 synonymous
    or missense row with a consistent ref_aa)."""
    seq = ["X"] * (n_residues + 1)  # 1-indexed; index 0 unused
    for pos, ref in zip(measured_df["position"], measured_df["ref_aa"]):
        if pd.notna(pos) and isinstance(ref, str):
            seq[int(pos)] = ref
    return seq


@dataclass
class Batch:
    aa_seq: torch.Tensor
    domain_seq: torch.Tensor
    coords: torch.Tensor
    mut_position: torch.Tensor
    aux_scores: torch.Tensor
    target: torch.Tensor


class ATMVariantDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        reference_seq: list[str],
        domain_track: np.ndarray,
        coord_track: np.ndarray,
        score_imputer: ScoreImputer,
        window_size: int | None = WINDOW_SIZE,
        has_target: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.reference_seq = reference_seq
        self.domain_track = domain_track
        self.coord_track = coord_track
        self.window_size = window_size
        self.has_target = has_target

        self.aux_scores = score_imputer.transform(self.df)
        if has_target:
            self.targets = arcsinh_transform(self.df["Combined_score"].to_numpy(dtype=np.float32))
        else:
            self.targets = np.zeros(len(self.df), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def _window_bounds(self, pos: int, n_residues: int) -> tuple[int, int, int]:
        """Return (start, end, local_index) 1-indexed inclusive-exclusive
        bounds and the mutated residue's 0-indexed offset within the crop."""
        if self.window_size is None:
            return 1, n_residues + 1, pos - 1
        half = self.window_size // 2
        start = max(1, pos - half)
        end = min(n_residues + 1, start + self.window_size)
        start = max(1, end - self.window_size)
        return start, end, pos - start

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        pos = int(row["position"])
        n_residues = len(self.reference_seq) - 1

        start, end, local_idx = self._window_bounds(pos, n_residues)

        seq = list(self.reference_seq[start:end])
        if 0 <= local_idx < len(seq):
            seq[local_idx] = row["alt_aa"] if isinstance(row["alt_aa"], str) else seq[local_idx]

        aa_ids = np.array([encode_amino_acid(a) for a in seq], dtype=np.int64)
        domain_ids = self.domain_track[start:end].astype(np.int64)
        coords = self.coord_track[start:end].astype(np.float32)

        return {
            "aa_seq": torch.from_numpy(aa_ids),
            "domain_seq": torch.from_numpy(domain_ids),
            "coords": torch.from_numpy(coords),
            "mut_position": torch.tensor(local_idx, dtype=torch.long),
            "aux_scores": torch.from_numpy(self.aux_scores[idx]),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
        }


def collate_batch(items: list[dict]) -> Batch:
    return Batch(
        aa_seq=torch.stack([it["aa_seq"] for it in items]),
        domain_seq=torch.stack([it["domain_seq"] for it in items]),
        coords=torch.stack([it["coords"] for it in items]),
        mut_position=torch.stack([it["mut_position"] for it in items]),
        aux_scores=torch.stack([it["aux_scores"] for it in items]),
        target=torch.stack([it["target"] for it in items]),
    )
