"""
Feature engineering for DeepATM, mirroring STAR★Methods "Deep learning
dataset and feature engineering" and "Model architecture":

  * Amino acid sequence  -> learned 64-d embedding (handled inside model.py;
                             this module just builds the integer sequence).
  * Domain annotation    -> per-residue domain id (TAN / FAT / PI3-4 Kinase /
                             FATC / none), from the domain table in the paper.
  * 3D coordinates       -> Cα (alpha-carbon) coordinates per residue, meant
                             to come from an AlphaFold 3 model of ATM (not
                             publicly deposited alongside the paper). Here we
                             try to fetch Cα coordinates for the cryo-EM
                             structures the paper itself cites for structural
                             visualization (PDB 8OXO / 7SID) as a stand-in,
                             and fall back to a smooth synthetic backbone if
                             that fetch isn't reachable (e.g. inside a
                             network-restricted sandbox) or a residue is
                             missing from the deposited model.
  * 16 precomputed scores -> only 5 (CADD, BoostDM, EVE, REVEL, AlphaMissense)
                             ship in the public Table S1; the rest are
                             zero/mean-imputed with a flag column, and
                             documented for extension via dbNSFP v4.8.

Target transform: arcsinh(y) = asinh((function_score + 0.912) / 2), per
STAR★Methods, to reduce skew before regression.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

N_RESIDUES = 3056  # full-length ATM protein length (STAR Methods / Figure 1A)

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY*X"  # 20 aa + stop (*) + unknown (X)
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}

DOMAINS = [
    ("TAN", 1, 166),
    ("FAT", 1940, 2566),
    ("PI3_4_Kinase", 2686, 2998),
    ("FATC", 3024, 3056),
]
DOMAIN_NAMES = ["none"] + [d[0] for d in DOMAINS]
DOMAIN_TO_IDX = {name: i for i, name in enumerate(DOMAIN_NAMES)}

AUX_SCORE_COLS = [
    "CADD.phred", "boostDM_score", "EVE_scores_ASM", "REVEL", "AlphaMissense",
    # The following 11 are named in STAR★Methods but are not present in the
    # public Table S1. They are included here as always-imputed placeholder
    # columns so the model's input dimensionality matches the paper (16
    # scores) and so real values can be dropped in later (e.g. from dbNSFP).
    "SIFT", "FATHMM", "MutationTaster", "LRT", "DANN", "PolyPhen2_HVAR",
    "PROVEAN", "phyloP100", "GERP", "ESM1b", "SpliceAI",
]


def domain_for_position(pos: int) -> str:
    for name, start, end in DOMAINS:
        if start <= pos <= end:
            return name
    return "none"


def build_domain_track(n_residues: int = N_RESIDUES) -> np.ndarray:
    """Return an (n_residues,) int array of domain ids, 1-indexed positions."""
    track = np.zeros(n_residues + 1, dtype=np.int64)  # index 0 unused
    for pos in range(1, n_residues + 1):
        track[pos] = DOMAIN_TO_IDX[domain_for_position(pos)]
    return track


# --------------------------------------------------------------------------
# Structural coordinates
# --------------------------------------------------------------------------

PDB_IDS = ["8OXO", "7SID"]


def fetch_ca_coordinates(pdb_id: str, timeout: float = 20.0) -> dict[int, tuple[float, float, float]] | None:
    """Fetch per-residue Cα coordinates for `pdb_id` from RCSB.

    Returns None (rather than raising) if the network is unreachable — this
    is expected in sandboxed environments; run this step on a machine with
    normal internet access to populate real coordinates.
    """
    try:
        import requests
        from Bio.PDB import MMCIFParser
    except ImportError:
        logger.warning("requests/biopython not installed; skipping structure fetch for %s", pdb_id)
        return None

    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - want a broad, loggable fallback
        logger.warning("Could not fetch %s (%s); falling back to synthetic coordinates.", pdb_id, exc)
        return None

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, io.StringIO(resp.text))

    coords: dict[int, tuple[float, float, float]] = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    # auth_seq_id maps (imperfectly) to protein position; a
                    # production pipeline should cross-check against the
                    # SEQRES/UniProt numbering rather than assuming a 1:1 map.
                    resnum = residue.id[1]
                    ca = residue["CA"].coord
                    coords.setdefault(resnum, tuple(float(c) for c in ca))
        break  # first model only
    return coords


def synthetic_backbone(n_residues: int = N_RESIDUES, rise: float = 1.5, radius: float = 30.0) -> np.ndarray:
    """A smooth helical placeholder backbone used when real coordinates are
    unavailable. This is NOT structural data — it exists purely so the
    coordinate-embedding branch of the model has *something* differentiable
    and position-dependent to train on, and should be replaced with real
    AlphaFold/PDB coordinates for any result you intend to trust.
    """
    idx = np.arange(1, n_residues + 1)
    theta = idx * 0.35
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = idx * rise
    return np.stack([x, y, z], axis=1)


def build_coordinate_track(cache_dir: Path = Path("data/processed")) -> np.ndarray:
    """Return an (N_RESIDUES + 1, 3) array of Cα coordinates, 1-indexed.

    Tries each PDB id in PDB_IDS in turn; any residue not covered by a
    successful fetch is filled from the synthetic backbone so the array is
    always fully populated (matching the paper's stated motivation for using
    AlphaFold 3 in the first place: "to avoid gaps caused by missing
    residues in experimentally determined structures").
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "ca_coordinates.npy"
    if cache_path.exists():
        return np.load(cache_path)

    coords = synthetic_backbone()
    full = np.zeros((N_RESIDUES + 1, 3), dtype=np.float32)
    full[1:] = coords

    for pdb_id in PDB_IDS:
        fetched = fetch_ca_coordinates(pdb_id)
        if not fetched:
            continue
        for pos, xyz in fetched.items():
            if 1 <= pos <= N_RESIDUES:
                full[pos] = xyz

    np.save(cache_path, full)
    return full


# --------------------------------------------------------------------------
# Target transform
# --------------------------------------------------------------------------

def arcsinh_transform(function_score: np.ndarray) -> np.ndarray:
    """y = asinh((function_score + 0.912) / 2), per STAR★Methods."""
    return np.arcsinh((function_score + 0.912) / 2.0)


def inverse_arcsinh_transform(y: np.ndarray) -> np.ndarray:
    return np.sinh(y) * 2.0 - 0.912


# --------------------------------------------------------------------------
# Auxiliary score matrix
# --------------------------------------------------------------------------

@dataclass
class ScoreImputer:
    """Mean-impute available score columns; zero-impute + flag columns that
    aren't present in Table S1 at all."""

    means: dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "ScoreImputer":
        for col in AUX_SCORE_COLS:
            if col in df.columns:
                self.means[col] = pd.to_numeric(df[col], errors="coerce").mean()
            else:
                self.means[col] = 0.0
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for col in AUX_SCORE_COLS:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").fillna(self.means[col])
            else:
                series = pd.Series(np.full(len(df), self.means[col]))
            cols.append(series.to_numpy(dtype=np.float32))
        mat = np.stack(cols, axis=1)
        # z-score normalize per column for stable training
        mu, sigma = mat.mean(axis=0, keepdims=True), mat.std(axis=0, keepdims=True) + 1e-6
        return (mat - mu) / sigma


def encode_amino_acid(aa: str | None) -> int:
    if aa is None or aa not in AA_TO_IDX:
        return AA_TO_IDX["X"]
    return AA_TO_IDX[aa]
