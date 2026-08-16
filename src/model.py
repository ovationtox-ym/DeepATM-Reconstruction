"""
DeepATM model architecture, per STAR★Methods "Model architecture" and
Figure 6A:

  * Amino acid embedding: 64-d, randomly initialized, one per residue in the
    ATM sequence.
  * Domain embedding: an additional embedding layer for domain annotation
    (TAN / FAT / PI3-4 Kinase / FATC / none) per residue.
  * Coordinate embedding: an MLP over each residue's 3D Cα coordinates
    (from AlphaFold 3 in the paper), integrated with the amino acid and
    domain embeddings.
  * Transformer encoder: 2 layers, 8 attention heads each, over the
    combined per-residue embedding sequence.

    Note on sequence length: the paper runs this over the entire 3,056-
    residue ATM sequence. Full self-attention over L=3,056 needs GPU-class
    memory (roughly L^2 x heads x batch floats per layer). This
    reconstruction supports that directly -- DeepATM(n_residues=3056) with a
    full-length sequence -- but also supports a cropped window centered on
    the mutated residue (see WINDOW_SIZE in train.py) so the pipeline is
    runnable end-to-end on a laptop/CPU. Use the full-length mode on a GPU
    for anything you want to compare against the paper's own numbers.
  * Fully connected head: the transformer's output at the mutated position
    is concatenated with 16 precomputed pathogenicity scores, then passed
    through a 128-unit ReLU layer and a single output neuron.

Loss: mean squared error against the arcsinh-transformed function score
(see features.arcsinh_transform).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .features import AA_ALPHABET, DOMAIN_NAMES, N_RESIDUES, AUX_SCORE_COLS

AA_VOCAB_SIZE = len(AA_ALPHABET)
DOMAIN_VOCAB_SIZE = len(DOMAIN_NAMES)
N_AUX_SCORES = len(AUX_SCORE_COLS)


class CoordinateMLP(nn.Module):
    """Projects raw (x, y, z) Cα coordinates into embedding space."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)


class DeepATM(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        n_transformer_layers: int = 2,
        n_heads: int = 8,
        n_aux_scores: int = N_AUX_SCORES,
        fc_hidden: int = 128,
        dropout: float = 0.1,
        n_residues: int = N_RESIDUES,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_residues = n_residues

        self.aa_embedding = nn.Embedding(AA_VOCAB_SIZE, embed_dim)
        self.domain_embedding = nn.Embedding(DOMAIN_VOCAB_SIZE, embed_dim)
        self.coord_mlp = CoordinateMLP(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim + n_aux_scores, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, 1),
        )

    def encode_sequence(
        self,
        aa_seq: torch.Tensor,      # (B, L) int64, mutant sequence per example
        domain_seq: torch.Tensor,  # (L,) or (B, L) int64, shared across batch if 1-D
        coords: torch.Tensor,      # (L, 3) or (B, L, 3) float32
    ) -> torch.Tensor:
        """Run the Transformer over a full-length sequence and return its
        per-position output, shape (B, L, embed_dim)."""
        if domain_seq.dim() == 1:
            domain_seq = domain_seq.unsqueeze(0).expand(aa_seq.size(0), -1)
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(aa_seq.size(0), -1, -1)

        x = self.aa_embedding(aa_seq) + self.domain_embedding(domain_seq) + self.coord_mlp(coords)
        return self.transformer(x)

    def forward(
        self,
        aa_seq: torch.Tensor,
        domain_seq: torch.Tensor,
        coords: torch.Tensor,
        mut_position: torch.Tensor,  # (B,) int64, 0-indexed position within aa_seq
        aux_scores: torch.Tensor,    # (B, n_aux_scores)
    ) -> torch.Tensor:
        encoded = self.encode_sequence(aa_seq, domain_seq, coords)  # (B, L, D)
        batch_idx = torch.arange(encoded.size(0), device=encoded.device)
        pos_repr = encoded[batch_idx, mut_position]  # (B, D)
        combined = torch.cat([pos_repr, aux_scores], dim=1)
        return self.fc(combined).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick shape sanity check with a small window (the full L=3056 sequence
    # needs GPU-class memory for self-attention -- see WINDOW_SIZE in
    # data_prep/train for the windowed-context mode used for CPU/smoke runs).
    model = DeepATM()
    print(f"DeepATM parameters: {count_parameters(model):,}")

    L = 256
    B = 4
    aa_seq = torch.randint(0, AA_VOCAB_SIZE, (B, L))
    domain_seq = torch.randint(0, DOMAIN_VOCAB_SIZE, (L,))
    coords = torch.randn(L, 3)
    mut_position = torch.randint(0, L, (B,))
    aux_scores = torch.randn(B, N_AUX_SCORES)

    out = model(aa_seq, domain_seq, coords, mut_position, aux_scores)
    print("output shape:", out.shape)
