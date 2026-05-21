from __future__ import annotations

import torch
from torch import nn


class MetadataConditioner(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_tokens: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size * num_tokens),
        )

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        tokens = self.net(metadata)
        return tokens.view(metadata.shape[0], self.num_tokens, self.hidden_size)


def min_snr_weights(timesteps: torch.Tensor, alphas_cumprod: torch.Tensor, gamma: float) -> torch.Tensor:
    if gamma <= 0:
        return torch.ones_like(timesteps, dtype=torch.float32)
    alpha = alphas_cumprod.to(timesteps.device)[timesteps]
    snr = alpha / (1.0 - alpha).clamp(min=1e-8)
    return torch.minimum(snr, torch.full_like(snr, gamma)) / snr.clamp(min=1e-8)
