"""
Feature Encoder - CNN for extracting acoustic features from raw waveforms.

Same architecture as wav2vec2/HuBERT for fair comparison:
- 7 convolutional layers
- Output: 512-dim features at 50Hz (20ms per frame for 16kHz audio)
"""

from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Single convolutional block with normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.GroupNorm(1, out_channels)  # LayerNorm equivalent
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class FeatureEncoder(nn.Module):
    """
    CNN Feature Encoder (wav2vec2-style).

    Converts raw waveform to frame-level features.

    Architecture:
    - 7 convolutional layers
    - Kernel sizes: [10, 3, 3, 3, 3, 2, 2]
    - Strides: [5, 2, 2, 2, 2, 2, 2]
    - Total downsampling: 320x (16kHz -> 50Hz)

    Args:
        in_channels: Input channels (1 for mono audio)
        conv_channels: Number of channels in conv layers
        embed_dim: Output embedding dimension
        kernel_sizes: List of kernel sizes for each layer
        strides: List of strides for each layer
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: int = 512,
        embed_dim: int = 768,
        kernel_sizes: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [10, 3, 3, 3, 3, 2, 2]
        if strides is None:
            strides = [5, 2, 2, 2, 2, 2, 2]

        assert len(kernel_sizes) == len(strides), "kernel_sizes and strides must have same length"

        self.conv_channels = conv_channels
        self.embed_dim = embed_dim

        # Build conv layers
        layers = []
        current_channels = in_channels

        for i, (k, s) in enumerate(zip(kernel_sizes, strides)):
            out_channels = conv_channels
            layers.append(ConvBlock(current_channels, out_channels, k, s))
            current_channels = out_channels

        self.conv_layers = nn.ModuleList(layers)

        # Projection to embedding dimension
        self.projection = nn.Linear(conv_channels, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Calculate receptive field
        self.total_stride = 1
        for s in strides:
            self.total_stride *= s

    def forward(
        self,
        waveform: torch.Tensor,
        return_all_layers: bool = False,
    ) -> torch.Tensor:
        """
        Extract features from waveform.

        Args:
            waveform: Raw audio (B, T) or (B, 1, T)
            return_all_layers: If True, return all intermediate representations

        Returns:
            features: Frame-level features (B, T', D) where T' = T // total_stride
        """
        # Ensure 3D input: (B, C, T)
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        x = waveform
        all_layers = []

        # Apply conv layers
        for conv in self.conv_layers:
            x = conv(x)
            if return_all_layers:
                all_layers.append(x)

        # Transpose: (B, C, T') -> (B, T', C)
        x = x.transpose(1, 2)

        # Project to embedding dimension
        x = self.projection(x)
        x = self.layer_norm(x)
        x = self.dropout(x)

        if return_all_layers:
            return x, all_layers

        return x

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input length."""
        length = input_length
        for conv in self.conv_layers:
            kernel_size = conv.conv.kernel_size[0]
            stride = conv.conv.stride[0]
            padding = conv.conv.padding[0]
            length = (length + 2 * padding - kernel_size) // stride + 1
        return length

    def get_receptive_field(self) -> int:
        """Calculate receptive field in samples."""
        rf = 1
        total_stride = 1
        for conv in self.conv_layers:
            kernel_size = conv.conv.kernel_size[0]
            stride = conv.conv.stride[0]
            rf = rf + (kernel_size - 1) * total_stride
            total_stride *= stride
        return rf


class FeatureEncoderSmall(FeatureEncoder):
    """Smaller feature encoder for small-scale experiments."""

    def __init__(
        self,
        embed_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__(
            conv_channels=256,
            embed_dim=embed_dim,
            kernel_sizes=[10, 3, 3, 3, 3],  # 5 layers instead of 7
            strides=[5, 2, 2, 2, 2],        # Less downsampling
            dropout=dropout,
        )
