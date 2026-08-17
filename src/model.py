"""
DeepATM model architecture, per STAR★Methods "Model architecture" and
Figure 6A:

  * Amino acid embedding: 64-d, randomly initialized, one per residue.
  * Domain embedding: an additional embedding layer for domain annotation
    (TAN / FAT / PI3-4 Kinase / FATC / none) per residue.
  * Coordinate embedding: an MLP over each residue's 3D Cα coordinates,
    "integrated with" the amino acid and domain embeddings. The paper does
    not say how they are combined; this reconstruction sums them (D7).
  * Transformer encoder: 2 layers, 8 attention heads each.
  * Fully connected head: the encoder output at the mutated position is
    concatenated with the precomputed scores, then passed through a 128-unit
    ReLU layer and a single output neuron.

Two documented departures from the literal text:

  * Positional encoding (D3). `nn.TransformerEncoder` has none. The coordinate
    branch does inject position-dependent signal, so the encoder is not fully
    permutation-equivariant, but it has no ordinal sense of the sequence. The
    paper is silent on this; sinusoidal encoding is added and can be disabled.
  * Sequence length. The paper runs over the full 3,056 residues. Attention
    alone at B=20, L=3056, 8 heads, fp32 is ~6 GB per layer, so this uses
    PyTorch's fused scaled_dot_product_attention (flash / memory-efficient
    backends) rather than the vanilla path.

Loss: mean squared error against the arcsinh-transformed function score.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .features import AA_ALPHABET, DOMAIN_NAMES, N_SCORES

AA_VOCAB_SIZE = len(AA_ALPHABET)
DOMAIN_VOCAB_SIZE = len(DOMAIN_NAMES)

# 16 scores + 16 missingness flags (features.ScoreImputer, with_flags=True).
N_AUX_SCORES = N_SCORES * 2

# x, y, z + an "is this residue resolved in the structure" channel.
COORD_INPUT_DIM = 4


class SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sinusoidal encoding, added to the residue embeddings."""

    def __init__(self, embed_dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        return x + self.pe[: x.size(1)].unsqueeze(0)


class CoordinateMLP(nn.Module):
    """Projects (x, y, z, resolved) into embedding space."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(COORD_INPUT_DIM, hidden_dim),
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
        use_positional_encoding: bool = True,
        use_coordinates: bool = True,
    ):
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})")

        self.embed_dim = embed_dim
        self.n_aux_scores = n_aux_scores
        self.use_coordinates = use_coordinates

        self.aa_embedding = nn.Embedding(AA_VOCAB_SIZE, embed_dim)
        self.domain_embedding = nn.Embedding(DOMAIN_VOCAB_SIZE, embed_dim)
        self.coord_mlp = CoordinateMLP(embed_dim) if use_coordinates else None
        self.positional = SinusoidalPositionalEncoding(embed_dim) if use_positional_encoding else None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim + n_aux_scores, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, 1),
        )

    def encode_sequence(
        self,
        aa_seq: torch.Tensor,      # (B, L) int64
        domain_seq: torch.Tensor,  # (L,) or (B, L) int64
        coords: torch.Tensor,      # (L, 4) or (B, L, 4) float32
    ) -> torch.Tensor:
        """Return the encoder's per-position output, (B, L, embed_dim)."""
        if domain_seq.dim() == 1:
            domain_seq = domain_seq.unsqueeze(0).expand(aa_seq.size(0), -1)
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(aa_seq.size(0), -1, -1)

        x = self.aa_embedding(aa_seq) + self.domain_embedding(domain_seq)
        if self.coord_mlp is not None:
            x = x + self.coord_mlp(coords)
        if self.positional is not None:
            x = self.positional(x)
        return self.transformer(x)

    def forward(
        self,
        aa_seq: torch.Tensor,
        domain_seq: torch.Tensor,
        coords: torch.Tensor,
        mut_position: torch.Tensor,  # (B,) int64, 0-indexed within aa_seq
        aux_scores: torch.Tensor,    # (B, n_aux_scores)
    ) -> torch.Tensor:
        if aux_scores.size(1) != self.n_aux_scores:
            raise ValueError(
                f"expected {self.n_aux_scores} auxiliary scores, got {aux_scores.size(1)}"
            )

        encoded = self.encode_sequence(aa_seq, domain_seq, coords)  # (B, L, D)
        batch_idx = torch.arange(encoded.size(0), device=encoded.device)
        pos_repr = encoded[batch_idx, mut_position]  # (B, D)
        combined = torch.cat([pos_repr, aux_scores.to(pos_repr.dtype)], dim=1)
        return self.fc(combined).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from .features import N_RESIDUES

    model = DeepATM()
    print(f"DeepATM parameters: {count_parameters(model):,}")

    for L, B in [(256, 4), (N_RESIDUES, 2)]:
        aa_seq = torch.randint(0, AA_VOCAB_SIZE, (B, L))
        domain_seq = torch.randint(0, DOMAIN_VOCAB_SIZE, (L,))
        coords = torch.randn(L, COORD_INPUT_DIM)
        mut_position = torch.randint(0, L, (B,))
        aux_scores = torch.randn(B, N_AUX_SCORES)
        out = model(aa_seq, domain_seq, coords, mut_position, aux_scores)
        print(f"  L={L:5d} B={B}: output {tuple(out.shape)}")
