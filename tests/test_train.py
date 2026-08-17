"""Training-loop checks (EXECUTION_PLAN.md M4 acceptance criteria).

Offline: no data files, no network, no model training.
"""
import numpy as np
import pandas as pd
import torch

from src.train import (
    LEARNING_RATE,
    RESTART_DECAY,
    ConsequenceBalancedSampler,
    WarmRestartSchedule,
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
