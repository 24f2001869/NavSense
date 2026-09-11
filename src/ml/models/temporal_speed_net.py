"""
Temporal Sequence Neural Network Architectures for Vehicle Speed Estimation.

Implements Branch B Sequence Models:
1. GRUSpeedNet: 2-layer Gated Recurrent Unit (Noise Compensation Network style, DVSE).
2. DilatedTCNNet: Causal Dilated 1D Convolutional Network with residual blocks.

Inputs:
- Tensor of shape (Batch, Timesteps=100, Channels=6) representing:
  [a_lin_x, a_lin_y, a_lin_z, gyro_yaw, gyro_pitch, gyro_roll]
Output:
- Scalar predicted instantaneous vehicle forward velocity v_t >= 0.0 (m/s).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUSpeedNet(nn.Module):
    """
    2-Layer GRU sequence model inspired by DVSE (arXiv:2505.18490).
    Input: (B, T, 6) -> Output: (B, 1) speed in m/s.
    """
    def __init__(self, in_channels: int = 6, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        proj = self.in_proj(x)
        out, _ = self.gru(proj)  # out: (B, T, hidden_dim)
        last_step = out[:, -1, :]  # (B, hidden_dim)
        speed = self.head(last_step)  # (B, 1)
        return F.relu(speed)  # Speeds strictly non-negative


class ChausalConv1dBlock(nn.Module):
    """Dilated Causal 1D Convolution Block with residual connection."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        self.res_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        # Causal padding on left
        padded1 = F.pad(x, (self.pad, 0))
        h = F.relu(self.norm1(self.conv1(padded1)))
        padded2 = F.pad(h, (self.pad, 0))
        h = self.norm2(self.conv2(padded2))
        res = self.res_proj(x)
        return F.relu(h + res)


class DilatedTCNNet(nn.Module):
    """
    Dilated Causal 1D-CNN (Temporal Convolutional Network) for Speed Estimation.
    Receptive field: 1 + 2 * (3-1) * (1 + 2 + 4 + 8) = 61 steps (6.1s).
    """
    def __init__(self, in_channels: int = 6, hidden_dim: int = 32):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.b1 = ChausalConv1dBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=1)
        self.b2 = ChausalConv1dBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2)
        self.b3 = ChausalConv1dBlock(hidden_dim, hidden_dim * 2, kernel_size=3, dilation=4)
        self.b4 = ChausalConv1dBlock(hidden_dim * 2, hidden_dim * 2, kernel_size=3, dilation=8)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> permute to (B, C, T)
        x_perm = x.permute(0, 2, 1)
        h = self.in_conv(x_perm)
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        h = self.b4(h)
        # Last timestep
        last_step = h[:, :, -1]  # (B, hidden_dim * 2)
        speed = self.head(last_step)
        return F.relu(speed)
