import torch
import torch.nn as nn


class ContinuousActor(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        h = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, h),
            nn.LayerNorm(h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.ELU(),
            nn.Linear(h, h // 2),
            nn.LayerNorm(h // 2),
            nn.ELU(),
            nn.Linear(h // 2, 2),
        )
        nn.init.constant_(self.net[-1].bias[0], 0.0)
        nn.init.constant_(self.net[-1].bias[1], 1.2)

    def forward(self, x):
        return self.net(x)


class TransformerActor(nn.Module):
    def __init__(self, state_dim=62, d_model=256, nhead=4,
                 num_layers=2, window_size=16, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.input_proj = nn.Linear(state_dim, d_model)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, window_size, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 2)
        nn.init.constant_(self.head.bias[0], 0.05)   # steer slight positive → break zero-steer local min
        nn.init.constant_(self.head.bias[1], 1.2)

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.head(x)


class ContinuousCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, output_dim: int = 1):
        super().__init__()
        h = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, h),
            nn.LayerNorm(h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.ELU(),
            nn.Linear(h, h // 2),
            nn.LayerNorm(h // 2),
            nn.ELU(),
            nn.Linear(h // 2, output_dim),
        )

    def forward(self, x):
        return self.net(x)
