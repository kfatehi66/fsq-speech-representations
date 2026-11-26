#!/usr/bin/env python3
"""
ProtoSpeech Hierarchical: Multi-Scale Prototype Learning

Instead of one prototype layer at the end, we have:
- Proto_L2: Prototypes for layer 2 (acoustic features - fricatives, plosives)
- Proto_L4: Prototypes for layer 4 (phonetic features - phoneme boundaries)
- Proto_L6: Prototypes for layer 6 (semantic features - context, coarticulation)

These are combined through linear projection for final representation.

Architecture:
  L2 hidden -> Proto_L2 (128 codes) -> assignments_L2
  L4 hidden -> Proto_L4 (128 codes) -> assignments_L4
  L6 hidden -> Proto_L6 (128 codes) -> assignments_L6
           |
  Concatenate [assign_L2, assign_L4, assign_L6]
           |
  Linear projection -> Final representation
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_encoder import FeatureEncoder
from .transformer import TransformerEncoder, TransformerLayer, RelativePositionalEncoding
from .prototype_layer import PrototypeLayer


class HierarchicalPrototypeLayer(nn.Module):
    """Single prototype layer for hierarchical extraction."""

    def __init__(
        self,
        embed_dim: int,
        num_prototypes: int,
        temperature: float = 0.1,
        use_projection: bool = True,
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.temperature = temperature

        # Learnable prototypes - initialized on unit hypersphere
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, embed_dim))
        nn.init.xavier_uniform_(self.prototypes)
        with torch.no_grad():
            self.prototypes.data = F.normalize(self.prototypes.data, dim=-1)

        # Optional projection layer
        if use_projection:
            self.projection = nn.Linear(embed_dim, embed_dim, bias=False)
        else:
            self.projection = nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute prototype similarities and assignments.

        Args:
            hidden_states: (B, T, D)

        Returns:
            Dictionary containing:
                - assignments: (B, T, num_prototypes) soft assignments
                - indices: (B, T) hard assignments
                - similarity: (B, T, num_prototypes) raw cosine similarities
                - prototype_embeddings: (B, T, D) prototype-weighted features
        """
        # Project hidden states
        h = self.projection(hidden_states)

        # Normalize for cosine similarity
        h_norm = F.normalize(h, dim=-1)  # (B, T, D)
        p_norm = F.normalize(self.prototypes, dim=-1)  # (K, D)

        # Cosine similarity: (B, T, K)
        similarity = torch.matmul(h_norm, p_norm.T)

        # Soft assignments via temperature-scaled softmax
        assignments = F.softmax(similarity / self.temperature, dim=-1)

        # Hard assignments (argmax)
        indices = similarity.argmax(dim=-1)

        # Prototype-weighted embeddings
        prototype_embeddings = torch.matmul(assignments, self.prototypes)

        return {
            'assignments': assignments,
            'indices': indices,
            'similarity': similarity,
            'prototype_embeddings': prototype_embeddings,
        }

    def get_prototype_stats(self) -> Dict[str, float]:
        """Get prototype diversity statistics."""
        p_norm = F.normalize(self.prototypes, dim=-1)
        sim_matrix = torch.matmul(p_norm, p_norm.T)

        # Exclude diagonal
        mask = ~torch.eye(self.num_prototypes, dtype=torch.bool, device=sim_matrix.device)
        off_diag = sim_matrix[mask]

        return {
            'num_prototypes': self.num_prototypes,
            'avg_similarity': off_diag.mean().item(),
            'max_similarity': off_diag.max().item(),
            'min_similarity': off_diag.min().item(),
        }


@dataclass
class HierarchicalProtoSpeechConfig:
    """Configuration for Hierarchical ProtoSpeech model."""

    # Model size
    name: str = "HierarchicalProtoSpeech-Small"

    # Feature encoder
    conv_channels: int = 512
    conv_kernel_sizes: List[int] = field(default_factory=lambda: [10, 3, 3, 3, 3, 2, 2])
    conv_strides: List[int] = field(default_factory=lambda: [5, 2, 2, 2, 2, 2, 2])

    # Transformer
    embed_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1

    # Hierarchical prototypes
    prototype_layers: List[int] = field(default_factory=lambda: [2, 4, 6])
    num_prototypes_per_layer: int = 128
    prototype_temperature: float = 0.1
    use_prototype_projection: bool = True

    # Output
    output_dim: int = 768  # Match HuBERT dim for distillation

    # Input
    sample_rate: int = 16000

    @classmethod
    def small(cls) -> "HierarchicalProtoSpeechConfig":
        """Small config for experiments (~35M params)."""
        return cls(
            name="HierarchicalProtoSpeech-Small",
            conv_channels=256,
            conv_kernel_sizes=[10, 3, 3, 3, 3, 2, 2],
            conv_strides=[5, 2, 2, 2, 2, 2, 2],
            embed_dim=512,
            num_layers=6,
            num_heads=8,
            ffn_dim=2048,
            prototype_layers=[2, 4, 6],
            num_prototypes_per_layer=128,
        )

    @classmethod
    def base(cls) -> "HierarchicalProtoSpeechConfig":
        """Base config (~95M params)."""
        return cls(
            name="HierarchicalProtoSpeech-Base",
            conv_channels=512,
            conv_kernel_sizes=[10, 3, 3, 3, 3, 2, 2],
            conv_strides=[5, 2, 2, 2, 2, 2, 2],
            embed_dim=768,
            num_layers=12,
            num_heads=12,
            ffn_dim=3072,
            prototype_layers=[3, 6, 9, 12],
            num_prototypes_per_layer=256,
        )


class HierarchicalProtoSpeech(nn.Module):
    """
    ProtoSpeech with hierarchical multi-scale prototypes.

    Extracts prototypes from multiple transformer layers (e.g., 2, 4, 6),
    then combines them through linear projection for the final representation.

    Hypothesis:
    - L2 prototypes = acoustic features (fricatives, plosives)
    - L4 prototypes = phonetic features (phoneme boundaries)
    - L6 prototypes = semantic features (context, coarticulation)

    Args:
        config: HierarchicalProtoSpeechConfig with model hyperparameters
    """

    def __init__(self, config: HierarchicalProtoSpeechConfig):
        super().__init__()

        self.config = config
        self.embed_dim = config.embed_dim
        self.prototype_layers = config.prototype_layers
        self.num_prototypes_per_layer = config.num_prototypes_per_layer

        # Validate prototype layers
        for layer_idx in config.prototype_layers:
            if layer_idx < 1 or layer_idx > config.num_layers:
                raise ValueError(
                    f"Prototype layer {layer_idx} out of range [1, {config.num_layers}]"
                )

        # Feature encoder (CNN)
        self.feature_encoder = FeatureEncoder(
            conv_channels=config.conv_channels,
            embed_dim=config.embed_dim,
            kernel_sizes=config.conv_kernel_sizes,
            strides=config.conv_strides,
            dropout=config.dropout,
        )

        # Positional encoding
        self.pos_encoding = RelativePositionalEncoding(config.embed_dim)

        # Individual transformer layers (for intermediate access)
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.embed_dim)

        # Hierarchical prototype layers
        self.proto_layers = nn.ModuleDict({
            f'proto_L{layer_idx}': HierarchicalPrototypeLayer(
                embed_dim=config.embed_dim,
                num_prototypes=config.num_prototypes_per_layer,
                temperature=config.prototype_temperature,
                use_projection=config.use_prototype_projection,
            )
            for layer_idx in config.prototype_layers
        })

        # Combiner: projects concatenated soft assignments to hidden dim
        total_proto_dim = config.num_prototypes_per_layer * len(config.prototype_layers)
        self.combiner = nn.Sequential(
            nn.Linear(total_proto_dim, config.embed_dim),
            nn.GELU(),
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim),
        )

        # Output projection (for distillation target matching, e.g., HuBERT 768-dim)
        self.output_proj = nn.Linear(config.embed_dim, config.output_dim)

        # Mask token for masked prediction
        self.mask_embedding = nn.Parameter(torch.zeros(config.embed_dim))
        nn.init.normal_(self.mask_embedding, std=0.02)

        self._print_config()

    def _print_config(self):
        """Print model configuration."""
        print(f"HierarchicalProtoSpeech initialized:")
        print(f"  Model: {self.config.name}")
        print(f"  Embed dim: {self.config.embed_dim}")
        print(f"  Transformer layers: {self.config.num_layers}")
        print(f"  Prototype layers: {self.prototype_layers}")
        print(f"  Prototypes per layer: {self.num_prototypes_per_layer}")
        print(f"  Total prototypes: {self.num_prototypes_per_layer * len(self.prototype_layers)}")
        print(f"  Output dim: {self.config.output_dim}")

    def forward(
        self,
        waveform: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_all_layers: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            waveform: (B, T) raw audio or (B, 1, T)
            mask: Optional boolean mask (B, T_frames) for masked prediction
            return_all_layers: whether to return intermediate representations

        Returns:
            Dictionary containing:
            - hidden_states: final combined representation (B, T, D)
            - output: projected output for distillation (B, T, output_dim)
            - combined_assignments: concatenated soft assignments (B, T, K*num_layers)
            - proto_L2, proto_L4, proto_L6: prototype outputs per layer
            - hidden_L2, hidden_L4, hidden_L6: raw hidden states per layer
            - extract_features: CNN features before transformer
        """
        # Extract acoustic features from CNN
        extract_features = self.feature_encoder(waveform)  # (B, T, D)

        # Apply mask if provided (for masked prediction)
        if mask is not None:
            masked_features = extract_features.clone()
            masked_features[mask] = self.mask_embedding
            hidden = masked_features
        else:
            hidden = extract_features

        # Add positional encoding
        hidden = self.pos_encoding(hidden)

        # Store intermediate representations
        layer_outputs = {}
        proto_outputs = {}

        # Pass through transformer layers, extracting at specified layers
        for layer_idx, transformer_layer in enumerate(self.transformer_layers, start=1):
            hidden, _ = transformer_layer(hidden)

            # Extract prototypes at specified layers
            if layer_idx in self.prototype_layers:
                proto_key = f'proto_L{layer_idx}'
                proto_out = self.proto_layers[proto_key](hidden)
                proto_outputs[proto_key] = proto_out
                layer_outputs[f'hidden_L{layer_idx}'] = hidden

        # Final layer norm
        hidden = self.final_norm(hidden)

        # Combine prototype assignments from all layers
        # Concatenate soft assignments: (B, T, K * num_proto_layers)
        all_assignments = torch.cat([
            proto_outputs[f'proto_L{layer_idx}']['assignments']
            for layer_idx in self.prototype_layers
        ], dim=-1)

        # Project combined assignments to hidden representation
        combined_hidden = self.combiner(all_assignments)  # (B, T, D)

        # Output projection (for distillation)
        output = self.output_proj(combined_hidden)  # (B, T, output_dim)

        return {
            'hidden_states': combined_hidden,
            'output': output,
            'combined_assignments': all_assignments,
            'transformer_final': hidden,
            'extract_features': extract_features,
            **proto_outputs,
            **layer_outputs,
        }

    def get_assignments(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get prototype assignments for audio at all layers.

        Args:
            waveform: Raw audio (B, T_audio)

        Returns:
            Dictionary with assignments from each prototype layer
        """
        with torch.no_grad():
            outputs = self.forward(waveform)
            return {
                f'assignments_L{layer_idx}': outputs[f'proto_L{layer_idx}']['assignments']
                for layer_idx in self.prototype_layers
            }

    def get_hard_assignments(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get hard (argmax) prototype assignments at all layers.

        Args:
            waveform: Raw audio (B, T_audio)

        Returns:
            Dictionary with hard indices from each prototype layer
        """
        with torch.no_grad():
            outputs = self.forward(waveform)
            return {
                f'indices_L{layer_idx}': outputs[f'proto_L{layer_idx}']['indices']
                for layer_idx in self.prototype_layers
            }

    def get_prototype_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each prototype layer."""
        stats = {}
        for layer_idx in self.prototype_layers:
            proto_key = f'proto_L{layer_idx}'
            stats[proto_key] = self.proto_layers[proto_key].get_prototype_stats()
        return stats

    def get_prototypes(self, layer_idx: Optional[int] = None) -> torch.Tensor:
        """
        Get prototype vectors.

        Args:
            layer_idx: Specific layer to get prototypes from. If None, returns all.

        Returns:
            Prototype tensor(s)
        """
        if layer_idx is not None:
            return self.proto_layers[f'proto_L{layer_idx}'].prototypes
        else:
            return {
                f'proto_L{idx}': self.proto_layers[f'proto_L{idx}'].prototypes
                for idx in self.prototype_layers
            }

    def encode(
        self,
        waveform: torch.Tensor,
        layer: int = -1,
    ) -> torch.Tensor:
        """
        Extract representations (for downstream tasks).

        Args:
            waveform: Raw audio (B, T_audio)
            layer: Which layer to extract from (-1 = combined output)
                   Use specific layer indices (2, 4, 6) for intermediate layers

        Returns:
            Representations (B, T, D)
        """
        with torch.no_grad():
            outputs = self.forward(waveform)

            if layer == -1:
                return outputs['hidden_states']
            elif layer in self.prototype_layers:
                return outputs[f'hidden_L{layer}']
            else:
                return outputs['transformer_final']

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in each component."""
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        params = {
            'feature_encoder': count(self.feature_encoder),
            'pos_encoding': count(self.pos_encoding),
            'transformer_layers': count(self.transformer_layers),
            'proto_layers': count(self.proto_layers),
            'combiner': count(self.combiner),
            'output_proj': count(self.output_proj),
            'mask_embedding': self.mask_embedding.numel(),
        }
        params['total'] = count(self)
        return params


class HierarchicalProtoSpeechSmall(HierarchicalProtoSpeech):
    """Small Hierarchical ProtoSpeech for experiments."""

    def __init__(self, num_prototypes_per_layer: int = 128):
        config = HierarchicalProtoSpeechConfig.small()
        config.num_prototypes_per_layer = num_prototypes_per_layer
        super().__init__(config)


def create_hierarchical_model(
    size: str = "small",
    num_prototypes_per_layer: int = 128,
    prototype_layers: Optional[List[int]] = None,
    **kwargs,
) -> HierarchicalProtoSpeech:
    """
    Create Hierarchical ProtoSpeech model.

    Args:
        size: Model size ("small", "base")
        num_prototypes_per_layer: Number of prototypes per layer
        prototype_layers: Which transformer layers to extract prototypes from
        **kwargs: Additional config overrides

    Returns:
        HierarchicalProtoSpeech model
    """
    if size == "small":
        config = HierarchicalProtoSpeechConfig.small()
    elif size == "base":
        config = HierarchicalProtoSpeechConfig.base()
    else:
        raise ValueError(f"Unknown size: {size}")

    config.num_prototypes_per_layer = num_prototypes_per_layer

    if prototype_layers is not None:
        config.prototype_layers = prototype_layers

    # Apply any overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return HierarchicalProtoSpeech(config)


# Quick test
if __name__ == '__main__':
    print("=" * 60)
    print("Testing HierarchicalProtoSpeech...")
    print("=" * 60)

    config = HierarchicalProtoSpeechConfig.small()
    model = HierarchicalProtoSpeech(config)

    # Test forward pass
    batch_size = 2
    audio_length = 16000 * 5  # 5 seconds
    waveform = torch.randn(batch_size, audio_length)

    print(f"\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(waveform)

    print(f"\n--- Shapes ---")
    print(f"Input waveform: {waveform.shape}")
    print(f"CNN features: {outputs['extract_features'].shape}")
    print(f"Combined hidden: {outputs['hidden_states'].shape}")
    print(f"Output (for distillation): {outputs['output'].shape}")
    print(f"Combined assignments: {outputs['combined_assignments'].shape}")

    for layer_idx in config.prototype_layers:
        proto_key = f'proto_L{layer_idx}'
        print(f"\n{proto_key}:")
        print(f"  Hidden: {outputs[f'hidden_L{layer_idx}'].shape}")
        print(f"  Assignments: {outputs[proto_key]['assignments'].shape}")
        print(f"  Indices: {outputs[proto_key]['indices'].shape}")

    # Prototype stats
    print("\n--- Prototype Statistics ---")
    stats = model.get_prototype_stats()
    for key, val in stats.items():
        print(f"{key}:")
        print(f"  num_prototypes: {val['num_prototypes']}")
        print(f"  avg_similarity: {val['avg_similarity']:.4f}")
        print(f"  max_similarity: {val['max_similarity']:.4f}")

    # Parameter count
    print("\n--- Parameter Count ---")
    params = model.count_parameters()
    for key, val in params.items():
        if key != 'total':
            print(f"  {key}: {val / 1e6:.2f}M")
    print(f"  TOTAL: {params['total'] / 1e6:.2f}M")

    print("\n" + "=" * 60)
    print("Test passed!")
    print("=" * 60)
