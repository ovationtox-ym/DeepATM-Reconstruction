"""
Feature engineering for DeepATM, mirroring STAR★Methods "Deep learning
dataset and feature engineering" and "Model architecture".

  * Amino acid sequence  -> learned 64-d embedding (in model.py); this module
                            supplies the wild-type sequence, fetched from
                            UniProt Q13315 rather than inferred from the
                            variant table.
  * Domain annotation    -> per-residue domain id (TAN / FAT / PI3-4 Kinase /
                            FATC / none), from the table in the paper.
  * 3D coordinates       -> Cα coordinates per residue. The paper used
                            AlphaFold 3, which is not deposited, and
                            AlphaFold DB has no ATM entry (Q13315 returns
                            404 — the protein is past the length cut-off).
                            We use PDB 7SID instead; see fetch_ca_coordinates
                            for why that specific entry.
  * 16 precomputed scores -> only 5 (CADD, BoostDM, EVE, REVEL, AlphaMissense)
                            ship in the public Table S1; the rest need
                            dbNSFP. Missing values are imputed with statistics
                            fitted on the training fold *only*, and every
                            column carries a missingness indicator.

Target transform: arcsinh(y) = asinh((function_score + 0.912) / 2), per
STAR★Methods, to reduce skew before regression.
"""
from __future__ import annotations

import gzip
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

N_RESIDUES = 3056  # full-length ATM protein length (STAR Methods / Figure 1A)
UNIPROT_ACC = "Q13315"

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY*X"  # 20 aa + stop (*) + unknown (X)
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}

# Verbatim from the domain table in STAR★Methods "Model architecture".
DOMAINS = [
    ("TAN", 1, 166),
    ("FAT", 1940, 2566),
    ("PI3_4_Kinase", 2686, 2998),
    ("FATC", 3024, 3056),
]
DOMAIN_NAMES = ["none"] + [d[0] for d in DOMAINS]
DOMAIN_TO_IDX = {name: i for i, name in enumerate(DOMAIN_NAMES)}

# The paper's 16 scores, in fixed order. The first five are present in
# Table S1; the rest are placeholders until a dbNSFP slice is joined in
# (EXECUTION_PLAN.md §2.1). Order must not change — it defines the input
# layout of the model's head.
AUX_SCORE_COLS = [
    "CADD.phred", "boostDM_score", "EVE_scores_ASM", "REVEL", "AlphaMissense",
    "SIFT", "FATHMM", "MutationTaster", "LRT", "DANN", "PolyPhen2_HVAR",
    "PROVEAN", "phyloP100", "GERP", "ESM1b", "SpliceAI",
]
N_SCORES = len(AUX_SCORE_COLS)

CACHE_DIR = Path("data/processed")


def _http_get(url: str, timeout: float = 180.0) -> bytes:
    """GET with the OS trust store, so TLS-inspecting proxies don't break fetches."""
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    import requests

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


# --------------------------------------------------------------------------
# Domain track
# --------------------------------------------------------------------------

def domain_for_position(pos: int) -> str:
    for name, start, end in DOMAINS:
        if start <= pos <= end:
            return name
    return "none"


def build_domain_track(n_residues: int = N_RESIDUES) -> np.ndarray:
    """Return an (n_residues + 1,) int array of domain ids, 1-indexed."""
    track = np.zeros(n_residues + 1, dtype=np.int64)  # index 0 unused
    for pos in range(1, n_residues + 1):
        track[pos] = DOMAIN_TO_IDX[domain_for_position(pos)]
    return track


# --------------------------------------------------------------------------
# Reference sequence
# --------------------------------------------------------------------------

def fetch_reference_sequence(cache_dir: Path = CACHE_DIR) -> str:
    """Return the wild-type ATM protein sequence (1-indexed via seq[pos - 1]).

    Fetched from UniProt, not reconstructed from the variant table. Inferring
    it from observed `ref_aa` values leaves any position absent from the frame
    as an unknown residue, which silently corrupts the input whenever the
    frame is a subset (e.g. a --max-rows smoke run).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"uniprot_{UNIPROT_ACC}.fasta"
    if cache_path.exists():
        text = cache_path.read_text()
    else:
        text = _http_get(f"https://rest.uniprot.org/uniprotkb/{UNIPROT_ACC}.fasta").decode()
        cache_path.write_text(text)

    seq = "".join(line for line in text.splitlines() if not line.startswith(">"))
    if len(seq) != N_RESIDUES:
        raise ValueError(f"Expected {N_RESIDUES} residues for {UNIPROT_ACC}, got {len(seq)}")
    return seq


# --------------------------------------------------------------------------
# Structural coordinates
# --------------------------------------------------------------------------

# PDB 7SID: cryo-EM ATM dimer, 2.53 A. Chosen over 8OXQ because its SEQRES is
# exactly the 3,056-residue ATM sequence, so mmCIF label_seq_id is the UniProt
# position with no offset — verified by matching all 2,773 modelled residues
# against the UniProt sequence (2,773/2,773 identical). 8OXQ is a 3,184-residue
# tagged construct whose label_seq_id does NOT align (91/1,448 identity), and
# would need an explicit SIFTS offset. 8OXO, used by an earlier version of this
# file, is a 12-residue synthetic peptide and not ATM at all.
#
# 7SID also contains nibrin (NBN) in chains B and D. The identity check below
# is what rejects them; do not rely on chain naming.
STRUCTURE_PDB_ID = "7SID"
MIN_IDENTITY = 0.95


def fetch_ca_coordinates(
    pdb_id: str = STRUCTURE_PDB_ID, reference_seq: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (coords, mask): an (N_RESIDUES + 1, 3) float array of Cα
    positions and an (N_RESIDUES + 1,) bool array marking which are real.

    Raises on any failure. There is deliberately no synthetic fallback — a
    fabricated backbone teaches the coordinate MLP a geometry that does not
    exist, and does so silently.
    """
    import gemmi

    if reference_seq is None:
        reference_seq = fetch_reference_sequence()

    raw = _http_get(f"https://files.rcsb.org/download/{pdb_id}.cif.gz")
    structure = gemmi.read_structure_string(gzip.decompress(raw).decode())
    structure.setup_entities()

    best: dict[int, tuple[float, float, float]] = {}
    best_identity = 0.0
    for chain in structure[0]:
        polymer = chain.get_polymer()
        if len(polymer) < N_RESIDUES // 2:
            continue  # e.g. the 10-residue NBN peptide in 7SID chains B/D

        matched = compared = 0
        coords: dict[int, tuple[float, float, float]] = {}
        for residue in polymer:
            pos = residue.label_seq  # SEQRES index == UniProt position for 7SID
            if pos is None or not 1 <= pos <= len(reference_seq):
                continue
            compared += 1
            info = gemmi.find_tabulated_residue(residue.name)
            if info and info.one_letter_code.upper() == reference_seq[pos - 1]:
                matched += 1
            ca = residue.find_atom("CA", "*")
            if ca is not None:
                coords[pos] = (ca.pos.x, ca.pos.y, ca.pos.z)

        identity = matched / compared if compared else 0.0
        if identity > best_identity:
            best_identity, best = identity, coords

    if best_identity < MIN_IDENTITY:
        raise ValueError(
            f"No chain in {pdb_id} matches the {UNIPROT_ACC} sequence "
            f"(best identity {best_identity:.1%} < {MIN_IDENTITY:.0%}). "
            f"The numbering assumption for this entry is wrong — check SIFTS."
        )

    coords = np.zeros((N_RESIDUES + 1, 3), dtype=np.float32)
    mask = np.zeros(N_RESIDUES + 1, dtype=bool)
    for pos, xyz in best.items():
        coords[pos] = xyz
        mask[pos] = True

    logger.info(
        "%s: %d/%d residues have Cα coordinates (%.1f%% identity)",
        pdb_id, int(mask[1:].sum()), N_RESIDUES, best_identity * 100,
    )
    return coords, mask


def normalize_coordinates(coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Center on the observed centroid and scale to unit RMS distance.

    Raw crystallographic coordinates are tens to hundreds of angstroms from
    the origin; feeding them to a small MLP unscaled makes the first layer's
    job needlessly hard. Masked positions are left at exactly zero, which is
    the centroid after centering and carries no positional information.
    """
    out = np.zeros_like(coords)
    observed = coords[mask]
    if len(observed) == 0:
        return out
    centroid = observed.mean(axis=0)
    scale = np.sqrt(((observed - centroid) ** 2).sum(axis=1).mean()) or 1.0
    out[mask] = (observed - centroid) / scale
    return out.astype(np.float32)


def build_coordinate_track(
    cache_dir: Path = CACHE_DIR, pdb_id: str = STRUCTURE_PDB_ID
) -> tuple[np.ndarray, np.ndarray]:
    """Return (normalized_coords, mask), cached on disk.

    The cache records which structure it came from, so a stale cache built
    from a different entry is detected rather than silently reused.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "ca_coordinates.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        if str(cached["pdb_id"]) == pdb_id:
            return cached["coords"], cached["mask"]
        logger.warning("Coordinate cache is from %s, refetching %s", cached["pdb_id"], pdb_id)

    raw_coords, mask = fetch_ca_coordinates(pdb_id)
    coords = normalize_coordinates(raw_coords, mask)
    np.savez(cache_path, coords=coords, mask=mask, pdb_id=pdb_id)
    return coords, mask


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
    """Impute and standardise the 16 auxiliary scores.

    All statistics are computed once in `fit` and reused by every `transform`.
    The previous implementation recomputed the mean and standard deviation
    from whatever frame it was handed, so the training and validation folds
    were standardised against different constants — both a train/test skew and
    a leak of validation-set statistics.

    `with_flags` appends a binary missingness indicator per score. Mean
    imputation alone is indistinguishable from a genuinely average score, and
    11 of the 16 columns are absent entirely while a further 23-39% of the
    five present ones are undefined for synonymous variants. The paper
    specifies 16 inputs and says nothing about imputation, so this is a
    documented deviation (D9); set with_flags=False for the literal 16.
    """

    with_flags: bool = True
    means: dict[str, float] = field(default_factory=dict)
    mu: np.ndarray | None = None
    sigma: np.ndarray | None = None

    @property
    def n_features(self) -> int:
        return N_SCORES * 2 if self.with_flags else N_SCORES

    def _raw_matrix(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (values, is_missing) before imputation, both (n, 16)."""
        values, missing = [], []
        for col in AUX_SCORE_COLS:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce")
            else:
                series = pd.Series(np.nan, index=df.index)
            missing.append(series.isna().to_numpy())
            values.append(series.to_numpy(dtype=np.float64))
        return np.stack(values, axis=1), np.stack(missing, axis=1)

    def fit(self, df: pd.DataFrame) -> "ScoreImputer":
        values, _ = self._raw_matrix(df)
        with warnings.catch_warnings():
            # Columns absent from Table S1 are all-NaN; nanmean warns, we mean 0.
            warnings.simplefilter("ignore", RuntimeWarning)
            col_means = np.nanmean(values, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        self.means = dict(zip(AUX_SCORE_COLS, col_means.tolist()))

        filled = np.where(np.isnan(values), col_means, values)
        self.mu = filled.mean(axis=0)
        self.sigma = filled.std(axis=0) + 1e-6
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.mu is None or self.sigma is None:
            raise RuntimeError("ScoreImputer.transform called before fit()")
        values, missing = self._raw_matrix(df)
        col_means = np.array([self.means[c] for c in AUX_SCORE_COLS])
        filled = np.where(np.isnan(values), col_means, values)
        standardised = (filled - self.mu) / self.sigma
        if self.with_flags:
            standardised = np.concatenate([standardised, missing.astype(np.float64)], axis=1)
        return standardised.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


def encode_amino_acid(aa: str | None) -> int:
    if aa is None or not isinstance(aa, str) or aa not in AA_TO_IDX:
        return AA_TO_IDX["X"]
    return AA_TO_IDX[aa]


def encode_sequence(seq: str) -> np.ndarray:
    """Encode a 1-indexed reference sequence into an (N_RESIDUES + 1,) int array."""
    ids = np.full(len(seq) + 1, AA_TO_IDX["X"], dtype=np.int64)
    for i, aa in enumerate(seq, start=1):
        ids[i] = encode_amino_acid(aa)
    return ids
