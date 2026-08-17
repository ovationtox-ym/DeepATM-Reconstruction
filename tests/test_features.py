"""Feature-engineering checks (EXECUTION_PLAN.md M2 acceptance criteria).

The score-matrix and imputer tests are offline. The coordinate test needs
`data/processed/ca_coordinates.npz`, built by the first run that touches
`build_coordinate_track()`; it skips if that cache is absent.
"""
import numpy as np
import pandas as pd
import pytest

from src.features import (
    AUX_SCORE_COLS,
    CACHE_DIR,
    N_RESIDUES,
    N_SCORES,
    ScoreImputer,
    normalize_coordinates,
)


def _frame(n: int, seed: int = 0) -> pd.DataFrame:
    """A frame with the five scores Table S1 actually ships, some missing."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        col: rng.normal(size=n) for col in AUX_SCORE_COLS[:5]
    })
    df.loc[df.index[: n // 3], "EVE_scores_ASM"] = np.nan  # undefined for synonymous
    return df


# --------------------------------------------------------------------------
# Auxiliary score matrix
# --------------------------------------------------------------------------

def test_score_matrix_shape_and_no_nan():
    imputer = ScoreImputer(with_flags=True)
    matrix = imputer.fit_transform(_frame(50))
    assert matrix.shape == (50, N_SCORES * 2)
    assert not np.isnan(matrix).any()


def test_score_matrix_literal_sixteen():
    """with_flags=False gives the paper's literal 16 inputs."""
    matrix = ScoreImputer(with_flags=False).fit_transform(_frame(20))
    assert matrix.shape == (20, N_SCORES) == (20, 16)


def test_missingness_flags_mark_the_missing_rows():
    df = _frame(30)
    matrix = ScoreImputer(with_flags=True).fit_transform(df)
    eve = AUX_SCORE_COLS.index("EVE_scores_ASM")
    flags = matrix[:, N_SCORES + eve]
    assert np.array_equal(flags.astype(bool), df["EVE_scores_ASM"].isna().to_numpy())


def test_absent_columns_are_flagged_missing_not_invented():
    """11 of the 16 scores are absent from Table S1 entirely. They must come
    through as zero-with-flag, never as a fabricated value."""
    matrix = ScoreImputer(with_flags=True).fit_transform(_frame(10))
    for col in AUX_SCORE_COLS[5:]:
        idx = AUX_SCORE_COLS.index(col)
        assert np.allclose(matrix[:, idx], 0.0)
        assert np.allclose(matrix[:, N_SCORES + idx], 1.0)


def test_transform_does_not_refit():
    """The defect this guards: transform() used to recompute mean and sigma
    from whatever frame it was handed, so train and val were standardised
    against different constants."""
    imputer = ScoreImputer().fit(_frame(200, seed=1))
    mu_before = imputer.mu.copy()

    # A validation fold with a deliberately different distribution.
    val = _frame(200, seed=2)
    val[AUX_SCORE_COLS[0]] += 100.0
    val_matrix = imputer.transform(val)

    assert np.allclose(imputer.mu, mu_before)
    # The shift must survive standardisation — if it were refit, it wouldn't.
    assert val_matrix[:, 0].mean() > 50


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        ScoreImputer().transform(_frame(5))


def test_fitted_imputer_is_reusable_across_folds():
    imputer = ScoreImputer().fit(_frame(100, seed=3))
    a = imputer.transform(_frame(10, seed=4))
    b = imputer.transform(_frame(10, seed=4))
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# Coordinates
# --------------------------------------------------------------------------

def test_normalize_coordinates_leaves_masked_at_zero():
    coords = np.random.default_rng(0).normal(size=(10, 3)) * 100 + 500
    mask = np.zeros(10, dtype=bool)
    mask[2:7] = True
    out = normalize_coordinates(coords, mask)
    assert np.allclose(out[~mask], 0.0)
    assert np.isclose(np.sqrt((out[mask] ** 2).sum(axis=1).mean()), 1.0, atol=1e-5)


@pytest.mark.parametrize("cache_name", ["ca_coordinates.npz"])
def test_consecutive_ca_distances_look_like_a_protein(cache_name):
    """A real backbone has consecutive Cα atoms ~3.8 Å apart. The deleted
    `synthetic_backbone()` produced an idealised helix; this is the check that
    would have caught it, and that catches a stale or wrong-chain fetch."""
    cache_path = CACHE_DIR / cache_name
    if not cache_path.exists():
        pytest.skip(f"{cache_path} not built yet")

    cached = np.load(cache_path)
    coords, mask = cached["coords"], cached["mask"]
    assert coords.shape == (N_RESIDUES + 1, 3)
    assert mask.shape == (N_RESIDUES + 1,)
    assert mask[1:].sum() > N_RESIDUES * 0.8, "too few residues resolved"

    def separations(k: int) -> np.ndarray:
        """Distances between residues k apart, for pairs both resolved."""
        both = mask[1:-k] & mask[1 + k:]
        return np.linalg.norm(coords[1 + k:][both] - coords[1:-k][both], axis=1)

    # Normalisation is a uniform rescale, so recover the angstrom scale from
    # the consecutive-Cα distance, which is ~3.8 A in every real backbone.
    scale = 3.8 / np.median(separations(1))
    steps = separations(1) * scale
    assert 3.6 < np.median(steps) < 4.0
    assert steps.std() < 0.3, "consecutive Ca spacing is too loose to be a real chain"

    # The discriminating test. i->i+4 is ~6.2 A and nearly constant in an
    # alpha helix, so the deleted `synthetic_backbone()` would produce almost
    # no spread here. A real fold mixes helix (~6 A) with strand and loop
    # (up to ~13 A), giving a broad, multimodal distribution.
    span4 = separations(4) * scale
    assert span4.std() > 1.0, "i->i+4 spacing is helix-uniform — synthetic geometry?"
    assert (span4 > 9.0).mean() > 0.05, "no extended (non-helical) segments at all"
