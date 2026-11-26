"""
ProtoSpeech: Interpretable Self-Supervised Speech Representations
through Prototype Learning

Main model class combining:
- Feature Encoder (CNN)
- Transformer Context Encoder
- Prototype Layer
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_encoder import FeatureEncoder, FeatureEncoderSmall
from .transformer import TransformerEncoder, TransformerEncoderSmall
from .prototype_layer import PrototypeLayer, GumbelPrototypeLayer
from .prototype_layer_ema import PrototypeLayerEMA


@dataclass
class ProtoSpeechConfig:
    """Configuration for ProtoSpeech model."""

    # Model size
    name: str = "ProtoSpeech-Base"

    # Feature encoder
    conv_channels: int = 512
    conv_kernel_sizes: List[int] = field(default_factory=lambda: [10, 3, 3, 3, 3, 2, 2])
    conv_strides: List[int] = field(default_factory=lambda: [5, 2, 2, 2, 2, 2, 2])

    # Transformer
    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ffn_dim: int = 3072
    dropout: float = 0.1

    # Prototype layer
    num_prototypes: int = 100
    prototype_temperature: float = 0.1
    use_prototype_projection: bool = True
    use_gumbel: bool = False
    use_sinkhorn: bool = False
    sinkhorn_iterations: int = 3

    # EMA prototype layer (VQ-VAE style, counters train/val collapse)
    use_ema_prototypes: bool = False
    ema_decay: float = 0.99
    commitment_weight: float = 0.25
    dead_threshold: int = 100

    # Input
    sample_rate: int = 16000

    @classmethod
    def small(cls) -> "ProtoSpeechConfig":
        """Small config for small-scale experiments (~30M params)."""
        # Note: 7 conv layers with total stride 320 (20 ms frames at 16 kHz),
        # the frame rate of HuBERT.
        return cls(
            name="ProtoSpeech-Small",
            conv_channels=256,
            conv_kernel_sizes=[10, 3, 3, 3, 3, 2, 2],  # 7 layers
            conv_strides=[5, 2, 2, 2, 2, 2, 2],        # Total stride 320
            embed_dim=512,
            num_layers=6,
            num_heads=8,
            ffn_dim=2048,
            num_prototypes=50,
        )

    @classmethod
    def base(cls) -> "ProtoSpeechConfig":
        """Base config (~95M params)."""
        return cls(
            name="ProtoSpeech-Base",
            embed_dim=768,
            num_layers=12,
            num_heads=12,
            ffn_dim=3072,
            num_prototypes=100,
        )

    @classmethod
    def large(cls) -> "ProtoSpeechConfig":
        """Large config (~300M params)."""
        return cls(
            name="ProtoSpeech-Large",
            embed_dim=1024,
            num_layers=24,
            num_heads=16,
            ffn_dim=4096,
            num_prototypes=200,
        )


class ProtoSpeech(nn.Module):
    """
    ProtoSpeech: Interpretable Self-Supervised Speech Model.

    Architecture:
    1. Feature Encoder: CNN extracts acoustic features from waveform
    2. Transformer Encoder: Contextualizes frame-level features
    3. Prototype Layer: Maps to learnable phonetic prototypes

    The prototype layer provides inherent interpretability:
    - Prototypes can be sonified (converted to audio)
    - Assignments reveal phonetic structure
    - "This sounds like prototype X" explanations

    Args:
        config: ProtoSpeechConfig with model hyperparameters
    """

    def __init__(self, config: ProtoSpeechConfig):
        super().__init__()

        self.config = config

        # Feature encoder (CNN)
        self.feature_encoder = FeatureEncoder(
            conv_channels=config.conv_channels,
            embed_dim=config.embed_dim,
            kernel_sizes=config.conv_kernel_sizes,
            strides=config.conv_strides,
            dropout=config.dropout,
        )

        # Transformer encoder
        self.transformer = TransformerEncoder(
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
        )

        # Prototype layer
        if config.use_ema_prototypes:
            # EMA-based prototype layer (VQ-VAE style), counters train/val collapse
            self.prototype_layer = PrototypeLayerEMA(
                embed_dim=config.embed_dim,
                num_prototypes=config.num_prototypes,
                temperature=config.prototype_temperature,
                ema_decay=config.ema_decay,
                commitment_weight=config.commitment_weight,
                dead_threshold=config.dead_threshold,
            )
        elif config.use_gumbel:
            self.prototype_layer = GumbelPrototypeLayer(
                embed_dim=config.embed_dim,
                num_prototypes=config.num_prototypes,
                temperature=config.prototype_temperature,
                use_projection=config.use_prototype_projection,
            )
        else:
            self.prototype_layer = PrototypeLayer(
                embed_dim=config.embed_dim,
                num_prototypes=config.num_prototypes,
                temperature=config.prototype_temperature,
                use_projection=config.use_prototype_projection,
                use_sinkhorn=config.use_sinkhorn,
                sinkhorn_iterations=config.sinkhorn_iterations,
            )

        # Mask token for masked prediction
        self.mask_embedding = nn.Parameter(torch.zeros(config.embed_dim))
        nn.init.normal_(self.mask_embedding, std=0.02)

    def forward(
        self,
        waveform: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_all_layers: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            waveform: Raw audio (B, T_audio) or (B, 1, T_audio)
            mask: Optional boolean mask (B, T_frames) for masked prediction
            return_all_layers: Return all transformer layer outputs

        Returns:
            Dictionary containing:
                - hidden_states: Transformer outputs (B, T, D)
                - similarities: Prototype similarities (B, T, P)
                - assignments: Soft prototype assignments (B, T, P)
                - prototype_embeddings: Prototype-weighted features (B, T, D)
                - hard_assignments: Argmax prototype indices (B, T)
                - extract_features: CNN features before transformer (B, T, D)
        """
        # Extract acoustic features
        extract_features = self.feature_encoder(waveform)  # (B, T, D)

        # Apply mask if provided (for masked prediction)
        if mask is not None:
            # Replace masked positions with learnable mask embedding
            masked_features = extract_features.clone()
            masked_features[mask] = self.mask_embedding
            transformer_input = masked_features
        else:
            transformer_input = extract_features

        # Contextualize with transformer
        hidden_states, all_layers, _ = self.transformer(
            transformer_input,
            return_all_layers=return_all_layers,
        )

        # Get prototype assignments
        proto_outputs = self.prototype_layer(hidden_states)

        # Build output dictionary
        outputs = {
            'hidden_states': hidden_states,
            'similarities': proto_outputs['similarities'],
            'assignments': proto_outputs['assignments'],
            'prototype_embeddings': proto_outputs['prototype_embeddings'],
            'hard_assignments': proto_outputs['hard_assignments'],
            'extract_features': extract_features,
        }

        # Pass through commitment loss if using EMA prototypes
        if 'commitment_loss' in proto_outputs:
            outputs['commitment_loss'] = proto_outputs['commitment_loss']

        if return_all_layers and all_layers is not None:
            outputs['all_layer_outputs'] = all_layers

        return outputs

    def get_prototypes(self) -> torch.Tensor:
        """Get prototype vectors."""
        return self.prototype_layer.prototypes

    def get_prototype_stats(self) -> Dict[str, float]:
        """Get prototype diversity statistics."""
        return self.prototype_layer.get_prototype_stats()

    def encode(
        self,
        waveform: torch.Tensor,
        layer: int = -1,
    ) -> torch.Tensor:
        """
        Extract representations (for downstream tasks).

        Args:
            waveform: Raw audio (B, T_audio)
            layer: Which transformer layer to extract from (-1 = last)

        Returns:
            Representations (B, T, D)
        """
        with torch.no_grad():
            outputs = self.forward(waveform, return_all_layers=True)

            if layer == -1 or 'all_layer_outputs' not in outputs:
                return outputs['hidden_states']
            else:
                return outputs['all_layer_outputs'][layer]

    def get_assignments(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get prototype assignments for audio.

        Args:
            waveform: Raw audio (B, T_audio)

        Returns:
            Soft assignments (B, T, P)
        """
        with torch.no_grad():
            outputs = self.forward(waveform)
            return outputs['assignments']

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in each component."""
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            'feature_encoder': count(self.feature_encoder),
            'transformer': count(self.transformer),
            'prototype_layer': count(self.prototype_layer),
            'mask_embedding': self.mask_embedding.numel(),
            'total': count(self),
        }


class ProtoSpeechSmall(ProtoSpeech):
    """Small ProtoSpeech for small-scale experiments."""

    def __init__(self, num_prototypes: int = 50):
        config = ProtoSpeechConfig.small()
        config.num_prototypes = num_prototypes
        super().__init__(config)


def create_model(
    size: str = "base",
    num_prototypes: int = 100,
    **kwargs,
) -> ProtoSpeech:
    """
    Create ProtoSpeech model.

    Args:
        size: Model size ("small", "base", "large")
        num_prototypes: Number of prototypes
        **kwargs: Additional config overrides

    Returns:
        ProtoSpeech model
    """
    if size == "small":
        config = ProtoSpeechConfig.small()
    elif size == "base":
        config = ProtoSpeechConfig.base()
    elif size == "large":
        config = ProtoSpeechConfig.large()
    else:
        raise ValueError(f"Unknown size: {size}")

    config.num_prototypes = num_prototypes

    # Apply any overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return ProtoSpeech(config)
