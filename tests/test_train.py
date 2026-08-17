"""Training-loop checks (EXECUTION_PLAN.md M4 acceptance criteria).

Offline: no data files and no network. The resume tests do train, but on
synthetic feature tracks and a handful of rows, so the whole file still runs
in seconds.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from src.dataset import FeatureTracks
from src.features import AUX_SCORE_COLS, N_RESIDUES
from src.train import (
    LEARNING_RATE,
    RESTART_DECAY,
    ConsequenceBalancedSampler,
    WarmRestartSchedule,
    load_resume_state,
    run_fingerprint,
    save_resume_state,
    train_one_fold,
)


# --------------------------------------------------------------------------
# Learning-rate schedule
# --------------------------------------------------------------------------

def test_restarts_land_where_the_paper_says():
    """T_0=10, T_mult=2 -> cycles start at epochs 0, 10, 30, 70, 150."""
    s = WarmRestartSchedule()
    starts = [e for e in range(151) if s.cycle_at(e)[1] == 0]
    assert starts == [0, 10, 30, 70, 150]


def test_each_restart_decays_the_peak_by_twenty_percent():
    s = WarmRestartSchedule()
    peaks = [s.lr_at(e) for e in (0, 10, 30, 70, 150)]
    expected = [LEARNING_RATE * RESTART_DECAY ** k for k in range(5)]
    assert np.allclose(peaks, expected)
    assert all(b < a for a, b in zip(peaks, peaks[1:]))


def test_lr_anneals_to_zero_within_each_cycle():
    s = WarmRestartSchedule()
    assert s.lr_at(9) < s.lr_at(0) * 0.05    # end of first cycle
    assert s.lr_at(29) < s.lr_at(10) * 0.05  # end of second


def test_decay_survives_the_optimizer_step():
    """The original defect: multiplying param_groups[...]["lr"] by 0.8 was
    undone the next time the scheduler recomputed lr from base_lrs. Applying
    the schedule must leave the *decayed* peak in the optimizer."""
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=LEARNING_RATE)
    s = WarmRestartSchedule()

    for epoch in range(31):
        s.apply(optimizer, epoch)
    assert np.isclose(optimizer.param_groups[0]["lr"], LEARNING_RATE * RESTART_DECAY ** 2)


# --------------------------------------------------------------------------
# Batch composition
# --------------------------------------------------------------------------

def _consequences(n_mis=100, n_syn=40, n_non=20) -> pd.Series:
    return pd.Series(["Missense"] * n_mis + ["Synonymous"] * n_syn + ["Nonsense"] * n_non)


def test_batch_is_eighteen_one_one():
    cons = _consequences()
    sampler = ConsequenceBalancedSampler(cons, batch_size=20, n_batches=50, seed=0)
    for batch in sampler:
        counts = cons.iloc[batch].value_counts()
        assert len(batch) == 20
        assert counts.get("Missense", 0) == 18
        assert counts.get("Synonymous", 0) == 1
        assert counts.get("Nonsense", 0) == 1


def test_sampler_length_matches_requested_batches():
    sampler = ConsequenceBalancedSampler(_consequences(), 20, n_batches=7)
    assert len(sampler) == 7
    assert sum(1 for _ in sampler) == 7


def test_missing_class_redistributes_rather_than_crashing():
    """--max-rows can produce a subset with no nonsense variants."""
    cons = pd.Series(["Missense"] * 30 + ["Synonymous"] * 5)
    sampler = ConsequenceBalancedSampler(cons, batch_size=20, n_batches=3, seed=0)
    for batch in sampler:
        counts = cons.iloc[batch].value_counts()
        assert counts.get("Nonsense", 0) == 0
        assert counts.get("Missense", 0) >= 18


def test_sampler_is_seeded():
    a = list(ConsequenceBalancedSampler(_consequences(), 20, 5, seed=7))
    b = list(ConsequenceBalancedSampler(_consequences(), 20, 5, seed=7))
    assert a == b


# --------------------------------------------------------------------------
# Resume — the property spot instances depend on
# --------------------------------------------------------------------------

WINDOW = 33


@pytest.fixture
def tracks():
    """Synthetic per-residue tracks, so no structure or sequence file is read."""
    rng = np.random.default_rng(0)
    return FeatureTracks(
        reference_ids=rng.integers(0, 20, size=N_RESIDUES + 1).astype(np.int64),
        domain_track=rng.integers(0, 4, size=N_RESIDUES + 1).astype(np.int64),
        coord_track=rng.normal(size=(N_RESIDUES + 1, 3)).astype(np.float32),
        coord_mask=np.ones(N_RESIDUES + 1, dtype=bool),
    )


def _variants(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    consequences = ["Missense"] * (n - 4) + ["Synonymous"] * 2 + ["Nonsense"] * 2
    df = pd.DataFrame({col: rng.normal(size=n) for col in AUX_SCORE_COLS[:5]})
    df["position"] = rng.integers(2, N_RESIDUES - 1, size=n)
    df["alt_aa"] = rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=n)
    df["Variant_consequence"] = consequences
    df["function_score"] = rng.normal(size=n)
    return df


def _fold(tracks, tmp_path, epochs, tag):
    return train_one_fold(
        _variants(40, seed=1), _variants(12, seed=2), tracks, torch.device("cpu"),
        epochs=epochs, batch_size=8, patience=99, fold_idx=0,
        window_size=WINDOW, seed=0,
        resume_path=tmp_path / f"resume_{tag}.pt",
        fingerprint=run_fingerprint(epochs=epochs, seed=0, window_size=WINDOW),
    )


def test_resumed_run_reproduces_an_uninterrupted_one(tracks, tmp_path, capsys):
    """The guarantee that makes spot instances usable: killing training and
    restarting it must give the same model, not merely a similar one.

    Equivalence alone would not prove much here — the fold re-seeds on entry,
    so a run that silently restarted from scratch would also match. The
    captured "resuming at epoch 3" is what rules that out.
    """
    uninterrupted = _fold(tracks, tmp_path, epochs=6, tag="whole")

    # The same run, stopped after 3 epochs and restarted — as a reclamation
    # does. The fingerprint is rewritten because only the epoch budget differs
    # between the two invocations; everything else about the fold is identical.
    _fold(tracks, tmp_path, epochs=3, tag="part")
    partial = torch.load(tmp_path / "resume_part.pt", weights_only=True)
    partial["fingerprint"] = run_fingerprint(epochs=6, seed=0, window_size=WINDOW)
    save_resume_state(tmp_path / "resume_part.pt", partial)
    capsys.readouterr()
    resumed = _fold(tracks, tmp_path, epochs=6, tag="part")

    assert "resuming at epoch 3" in capsys.readouterr().out
    assert np.allclose(uninterrupted["val_preds"], resumed["val_preds"])
    assert uninterrupted["val_loss"] == pytest.approx(resumed["val_loss"])
    assert uninterrupted["best_epoch"] == resumed["best_epoch"]


def test_resume_file_is_written_every_epoch(tracks, tmp_path):
    _fold(tracks, tmp_path, epochs=2, tag="w")
    state = torch.load(tmp_path / "resume_w.pt", weights_only=True)
    assert state["next_epoch"] == 2
    assert state["done"] is False


def test_resume_state_loads_without_arbitrary_unpickling(tmp_path):
    """Everything stored must be a tensor, a primitive, or a container of
    those, so the file reads back under weights_only=True."""
    fingerprint = run_fingerprint(epochs=1, seed=0)
    save_resume_state(tmp_path / "s.pt", {"fingerprint": fingerprint, "next_epoch": 1})
    assert load_resume_state(tmp_path / "s.pt", fingerprint)["next_epoch"] == 1


def test_mismatched_fingerprint_refuses_to_resume(tmp_path):
    """Resuming into a checkpoint trained under other settings would produce a
    model matching neither configuration. It must fail loudly, not restart."""
    save_resume_state(tmp_path / "s.pt", {"fingerprint": run_fingerprint(epochs=10)})
    with pytest.raises(RuntimeError, match="different settings"):
        load_resume_state(tmp_path / "s.pt", run_fingerprint(epochs=150))


def test_absent_resume_file_is_not_an_error(tmp_path):
    assert load_resume_state(tmp_path / "nope.pt", run_fingerprint(epochs=1)) is None
