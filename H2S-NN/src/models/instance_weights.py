from __future__ import annotations

import torch
from torch import nn


class InstanceWeightNetwork(nn.Module):
    """Positive, neutral-initialized weight for one concrete path instance."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        output = self.net[-1]
        if not isinstance(output, nn.Linear):
            raise TypeError("InstanceWeightNetwork output layer must be linear.")
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, instance_input: torch.Tensor) -> torch.Tensor:
        """Return one learned weight in ``(0, 2)`` per input row."""
        logits = self.net(instance_input).squeeze(-1)
        return 2.0 * torch.sigmoid(logits)
