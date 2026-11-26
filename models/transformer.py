"""
Transformer Encoder for contextualizing frame-level features.

Standard transformer with:
- Multi-head self-attention
- Feedforward layers
- Relative positional encoding (like wav2vec2)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativePositionalEncoding(nn.Module):
    """
    Relative Positional Encoding.

    Adds relative position information to attention scores.
    Similar to wav2vec2's convolutional position encoding.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        kernel_size: int = 128,
        groups: int = 16,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional information.

        Args:
            x: Input tensor (B, T, D)

        Returns:
            x: Tensor with positional encoding (B, T, D)
        """
        # Conv expects (B, D, T)
        residual = x
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        # Remove extra frame from padding
        x = x[:, :residual.size(1), :]

        x = self.norm(x)
        return residual + x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional relative positions."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply multi-head attention.

        Args:
            x: Input tensor (B, T, D)
            attention_mask: Optional mask (B, T) or (B, T, T)
            return_attention: Whether to return attention weights

        Returns:
            output: Attended tensor (B, T, D)
            attention: Optional attention weights (B, H, T, T)
        """
        B, T, D = x.shape

        # Project queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, H, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: (B, H, T, T)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # (B, T) -> (B, 1, 1, T)
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                # (B, T, T) -> (B, 1, T, T)
                attention_mask = attention_mask.unsqueeze(1)
            attn = attn.masked_fill(attention_mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        output = torch.matmul(attn, v)  # (B, H, T, head_dim)

        # Reshape back to (B, T, D)
        output = output.transpose(1, 2).contiguous().view(B, T, D)
        output = self.out_proj(output)

        if return_attention:
            return output, attn
        return output, None


class FeedForward(nn.Module):
    """Feedforward network with GELU activation."""

    def __init__(
        self,
        embed_dim: int = 768,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerLayer(nn.Module):
    """Single transformer layer."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ffn = FeedForward(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply transformer layer.

        Args:
            x: Input tensor (B, T, D)
            attention_mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            output: Transformed tensor (B, T, D)
            attention: Optional attention weights
        """
        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)
        x, attn = self.self_attn(x, attention_mask, return_attention)
        x = self.dropout(x)
        x = residual + x

        # Pre-norm feedforward
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, attn


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder for speech representation.

    Args:
        embed_dim: Embedding dimension (default: 768)
        num_layers: Number of transformer layers (default: 12)
        num_heads: Number of attention heads (default: 12)
        ffn_dim: Feedforward dimension (default: 3072)
        dropout: Dropout probability (default: 0.1)
        use_relative_pos: Use relative positional encoding
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        use_relative_pos: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Positional encoding
        if use_relative_pos:
            self.pos_encoding = RelativePositionalEncoding(embed_dim)
        else:
            self.pos_encoding = None

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_layers: bool = False,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list], Optional[list]]:
        """
        Apply transformer encoder.

        Args:
            x: Input features (B, T, D)
            attention_mask: Optional attention mask
            return_all_layers: Return all layer outputs
            return_attention: Return attention weights

        Returns:
            output: Final layer output (B, T, D)
            all_layers: Optional list of all layer outputs
            all_attentions: Optional list of attention weights
        """
        # Add positional encoding
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        all_layers = [] if return_all_layers else None
        all_attentions = [] if return_attention else None

        for layer in self.layers:
            x, attn = layer(x, attention_mask, return_attention)

            if return_all_layers:
                all_layers.append(x)
            if return_attention:
                all_attentions.append(attn)

        x = self.final_norm(x)

        return x, all_layers, all_attentions


class TransformerEncoderSmall(TransformerEncoder):
    """Smaller transformer for small-scale experiments."""

    def __init__(
        self,
        embed_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_layers=6,
            num_heads=8,
            ffn_dim=2048,
            dropout=dropout,
        )
