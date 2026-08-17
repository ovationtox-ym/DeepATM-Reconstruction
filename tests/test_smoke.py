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
    encode_sequence,
    inverse_arcsinh_transform,
)
from src.model import N_AUX_SCORES, DeepATM, count_parameters


def test_domain_track_shape():
    track = build_domain_track()
    assert track.shape == (N_RESIDUES + 1,)
    assert track.min() >= 0


def test_domain_track_boundaries():
    """Domain spans are verbatim from the paper; guard them against edits."""
    track = build_domain_track()
    assert track[1] == track[166] != 0        # TAN
    assert track[167] == 0
    assert track[1940] == track[2566] != 0    # FAT
    assert track[3024] == track[3056] != 0    # FATC
    assert track[3023] == 0


def test_arcsinh_roundtrip():
    x = np.array([-5.0, -0.912, 0.0, 2.5])
    y = arcsinh_transform(x)
    x_hat = inverse_arcsinh_transform(y)
    assert np.allclose(x, x_hat, atol=1e-4)


def test_encode_amino_acid_unknown():
    assert encode_amino_acid(None) == encode_amino_acid("X")
    assert encode_amino_acid("garbage") == encode_amino_acid("X")


def test_encode_sequence_is_one_indexed():
    ids = encode_sequence("MAG")
    assert len(ids) == 4
    assert ids[1] == encode_amino_acid("M")
    assert ids[3] == encode_amino_acid("G")


def test_model_forward_shape():
    model = DeepATM()
    assert count_parameters(model) > 0

    L, B = 64, 3
    aa_seq = torch.randint(0, 22, (B, L))
    domain_seq = torch.randint(0, 5, (L,))
    coords = torch.randn(L, 4)
    mut_position = torch.randint(0, L, (B,))
    aux_scores = torch.randn(B, N_AUX_SCORES)

    out = model(aa_seq, domain_seq, coords, mut_position, aux_scores)
    assert out.shape == (B,)


def test_model_rejects_wrong_aux_width():
    model = DeepATM()
    L, B = 16, 2
    with __import__("pytest").raises(ValueError):
        model(
            torch.randint(0, 22, (B, L)),
            torch.zeros(L, dtype=torch.long),
            torch.zeros(L, 4),
            torch.zeros(B, dtype=torch.long),
            torch.randn(B, N_AUX_SCORES - 1),
        )


def test_model_head_reads_the_mutated_position():
    """The head must depend on the encoder output *at* the mutation, not on a
    pooled representation — otherwise two variants at different positions with
    identical scores would be indistinguishable."""
    torch.manual_seed(0)
    model = DeepATM().eval()
    L = 32
    aa_seq = torch.randint(0, 20, (2, L))
    domain_seq = torch.zeros(L, dtype=torch.long)
    coords = torch.randn(L, 4)
    aux = torch.zeros(2, N_AUX_SCORES)

    with torch.no_grad():
        out = model(aa_seq, domain_seq, coords, torch.tensor([3, 3]), aux)
        out_moved = model(aa_seq, domain_seq, coords, torch.tensor([3, 17]), aux)
    assert torch.isclose(out[0], out_moved[0])
    assert not torch.isclose(out[1], out_moved[1])
