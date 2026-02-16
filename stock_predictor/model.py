from __future__ import annotations

import torch
import torch.nn as nn


class MultiAssetTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_symbols: int,
        n_horizons: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_mult: int,
        dropout: float,
        max_seq_len: int,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.symbol_emb = nn.Embedding(n_symbols, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_horizons),
        )

    def forward(self, x: torch.Tensor, symbol_idx: torch.Tensor) -> torch.Tensor:
        _, t, _ = x.shape
        h = self.input_proj(x)
        h = h + self.pos_emb[:, :t, :]

        # Add symbol context at final token only.
        sym = self.symbol_emb(symbol_idx).unsqueeze(1)
        h[:, -1:, :] = h[:, -1:, :] + sym

        h = self.encoder(h)
        return self.head(self.norm(h[:, -1, :]))
