"""PyTorch policy/value network with LSTM for forex RL trading."""

from __future__ import annotations

import torch
import torch.nn as nn


class TradingLSTMPolicy(nn.Module):
    """LSTM-based actor-critic network for PPO."""

    def __init__(
        self,
        input_size: int,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 1,
        fc_size: int = 128,
        num_actions: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.shared = nn.Sequential(
            nn.Linear(lstm_hidden_size, fc_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.actor_head = nn.Linear(fc_size, num_actions)
        self.critic_head = nn.Linear(fc_size, 1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape [batch, sequence_length, feature_count].
        Returns:
            action_probs: Probability distribution over 3 discrete actions.
            state_value: Estimated value function for PPO critic.
            logits: Raw action logits (used for stable sampling/loss).
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]

        features = self.shared(last_hidden)
        logits = self.actor_head(features)
        action_probs = torch.softmax(logits, dim=-1)
        state_value = self.critic_head(features).squeeze(-1)

        return action_probs, state_value, logits
