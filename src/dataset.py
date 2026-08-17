"""
PyTorch Dataset that turns rows of the processed variant tables into
model-ready tensors: the mutant amino-acid sequence, the matching
domain/coordinate tracks, the mutated position, and the auxiliary score
vector.

STAR★Methods trains DeepATM over the full 3,056-residue sequence. The window
is configurable so the same code can run in full-length mode (the paper's
setting, needs a GPU) or in a laptop-friendly cropped mode. A windowed run is
NOT the paper's model — report the window size with any result.
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
    encode_amino_acid,
)

WINDOW_SIZE = 255  # odd -> symmetric window around the mutated residue; None = full length

TARGET_COLUMN = "function_score"


@dataclass
class Batch:
    aa_seq: torch.Tensor
    domain_seq: torch.Tensor
    coords: torch.Tensor
    mut_position: torch.Tensor
    aux_scores: torch.Tensor
    target: torch.Tensor


class ATMVariantDataset(Dataset):
    """One variant per item.

    `aux_matrix` is precomputed by a `ScoreImputer` that was fitted on the
    training fold, and passed in rather than derived here — the dataset must
    not be able to fit statistics on its own rows.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        reference_ids: np.ndarray,
        domain_track: np.ndarray,
        coord_track: np.ndarray,
        coord_mask: np.ndarray,
        aux_matrix: np.ndarray,
        window_size: int | None = WINDOW_SIZE,
        has_target: bool = True,
        target_column: str = TARGET_COLUMN,
    ):
        self.df = df.reset_index(drop=True)
        self.reference_ids = reference_ids
        self.domain_track = domain_track
        self.window_size = window_size

        # Coordinates carry a 4th channel flagging whether the position was
        # resolved in the structure, so the MLP can learn a distinct response
        # for unmodelled residues instead of reading the zero-fill as a real
        # location at the centroid.
        self.coord_track = np.concatenate(
            [coord_track, coord_mask.astype(np.float32)[:, None]], axis=1
        ).astype(np.float32)

        if len(aux_matrix) != len(self.df):
            raise ValueError(
                f"aux_matrix has {len(aux_matrix)} rows, dataframe has {len(self.df)}"
            )
        self.aux_scores = aux_matrix.astype(np.float32)

        self.positions = self.df["position"].to_numpy(dtype=np.int64)
        self.alt_aa = self.df["alt_aa"].tolist()

        if has_target:
            scores = self.df[target_column].to_numpy(dtype=np.float32)
            if np.isnan(scores).any():
                raise ValueError(
                    f"{int(np.isnan(scores).sum())} rows have no {target_column}; "
                    "filter them before constructing the dataset"
                )
            self.targets = arcsinh_transform(scores).astype(np.float32)
        else:
            self.targets = np.zeros(len(self.df), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def _window_bounds(self, pos: int) -> tuple[int, int, int]:
        """Return 1-indexed [start, end) bounds and the mutated residue's
        0-indexed offset within the crop."""
        if self.window_size is None:
            return 1, N_RESIDUES + 1, pos - 1
        half = self.window_size // 2
        start = max(1, pos - half)
        end = min(N_RESIDUES + 1, start + self.window_size)
        start = max(1, end - self.window_size)
        return start, end, pos - start

    def __getitem__(self, idx: int):
        pos = int(self.positions[idx])
        start, end, local_idx = self._window_bounds(pos)

        aa_ids = self.reference_ids[start:end].copy()
        alt = self.alt_aa[idx]
        if 0 <= local_idx < len(aa_ids) and isinstance(alt, str):
            aa_ids[local_idx] = encode_amino_acid(alt)

        return {
            "aa_seq": torch.from_numpy(aa_ids),
            "domain_seq": torch.from_numpy(self.domain_track[start:end].copy()),
            "coords": torch.from_numpy(self.coord_track[start:end].copy()),
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


def build_dataset(
    df: pd.DataFrame,
    tracks: "FeatureTracks",
    imputer: ScoreImputer,
    window_size: int | None = WINDOW_SIZE,
    has_target: bool = True,
) -> ATMVariantDataset:
    """Convenience constructor binding a frame to shared tracks and a fitted imputer."""
    return ATMVariantDataset(
        df,
        tracks.reference_ids,
        tracks.domain_track,
        tracks.coord_track,
        tracks.coord_mask,
        imputer.transform(df),
        window_size=window_size,
        has_target=has_target,
    )


@dataclass
class FeatureTracks:
    """The per-residue inputs shared by every variant: sequence, domains, structure."""

    reference_ids: np.ndarray
    domain_track: np.ndarray
    coord_track: np.ndarray
    coord_mask: np.ndarray

    @classmethod
    def load(cls) -> "FeatureTracks":
        from .features import (
            build_coordinate_track,
            build_domain_track,
            encode_sequence,
            fetch_reference_sequence,
        )

        coords, mask = build_coordinate_track()
        return cls(
            reference_ids=encode_sequence(fetch_reference_sequence()),
            domain_track=build_domain_track(),
            coord_track=coords,
            coord_mask=mask,
        )
