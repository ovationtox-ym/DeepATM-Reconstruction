"""Fast sanity checks that don't require the supplementary data files.

Run with: pytest tests/
"""
import numpy as np
import torch

from src.features import (
    N_RESIDUES,
    arcsinh_transform,
    build_domain_track,
    encode_amino_acid,
    inverse_arcsinh_transform,
)
from src.model import DeepATM, count_parameters


def test_domain_track_shape():
    track = build_domain_track()
    assert track.shape == (N_RESIDUES + 1,)
    assert track.min() >= 0


def test_arcsinh_roundtrip():
    x = np.array([-5.0, -0.912, 0.0, 2.5])
    y = arcsinh_transform(x)
    x_hat = inverse_arcsinh_transform(y)
    assert np.allclose(x, x_hat, atol=1e-4)


def test_encode_amino_acid_unknown():
    assert encode_amino_acid(None) == encode_amino_acid("X")
    assert encode_amino_acid("garbage") == encode_amino_acid("X")


def test_model_forward_shape():
    model = DeepATM()
    assert count_parameters(model) > 0

    L, B = 64, 3
    aa_seq = torch.randint(0, 22, (B, L))
    domain_seq = torch.randint(0, 5, (L,))
    coords = torch.randn(L, 3)
    mut_position = torch.randint(0, L, (B,))
    aux_scores = torch.randn(B, 16)

    out = model(aa_seq, domain_seq, coords, mut_position, aux_scores)
    assert out.shape == (B,)
