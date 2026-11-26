"""
Combined ProtoSpeech Loss

Orchestrates all loss components:
L_total = L_masked + λ₁·L_prototype + λ₂·L_diversity + λ₃·L_contrastive
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn

from .diversity_loss import DiversityLoss
from .prototype_loss import PrototypeLoss
from .masked_prediction_loss import MaskedPrototypePredictionLoss, SpanMaskGenerator
from .contrastive_loss import ContrastiveLoss


@dataclass
class LossConfig:
    """Configuration for ProtoSpeech loss."""
    # Loss weights
    lambda_masked: float = 1.0
    lambda_prototype: float = 1.0
    lambda_diversity: float = 0.5
    lambda_contrastive: float = 1.0

    # Prototype loss
    lambda_cluster: float = 1.0
    lambda_separation: float = 0.5
    separation_margin: float = 0.5

    # Diversity loss
    lambda_entropy: float = 1.0
    lambda_orthogonality: float = 0.1
    lambda_temporal: float = 0.5
    target_entropy_ratio: float = 0.9
    min_temporal_variance: float = 0.1

    # Masking
    mask_prob: float = 0.15
    mask_span_length: int = 10

    # Temperature
    temperature: float = 0.1

    # Number of prototypes (needed for entropy loss)
    num_prototypes: int = 100


class ProtoSpeechLoss(nn.Module):
    """
    Combined ProtoSpeech Training Loss.

    Combines:
    1. Masked Prototype Prediction (main SSL objective)
    2. Prototype Loss (cluster + separation)
    3. Diversity Loss (prevent collapse)
    4. Contrastive Loss (view invariance, optional)

    Args:
        config: LossConfig with all hyperparameters
    """

    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config

        # Loss components
        self.masked_loss = MaskedPrototypePredictionLoss(
            temperature=config.temperature,
        )

        self.prototype_loss = PrototypeLoss(
            lambda_cluster=config.lambda_cluster,
            lambda_separation=config.lambda_separation,
            separation_margin=config.separation_margin,
        )

        self.diversity_loss = DiversityLoss(
            num_prototypes=config.num_prototypes,
            lambda_entropy=config.lambda_entropy,
            lambda_orthogonality=config.lambda_orthogonality,
            lambda_temporal=config.lambda_temporal,
            target_entropy_ratio=config.target_entropy_ratio,
            min_temporal_variance=config.min_temporal_variance,
        )

        self.contrastive_loss = ContrastiveLoss(
            temperature=config.temperature,
        )

        # Mask generator
        self.mask_generator = SpanMaskGenerator(
            mask_prob=config.mask_prob,
            mask_span_length=config.mask_span_length,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prototypes: torch.Tensor,
        assignments: torch.Tensor,
        similarities: torch.Tensor,
        hidden_states_aug: Optional[torch.Tensor] = None,
        assignments_aug: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        target_assignments: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined ProtoSpeech loss.

        Args:
            hidden_states: Transformer outputs (B, T, D)
            prototypes: Prototype vectors (P, D)
            assignments: Soft prototype assignments (B, T, P)
            similarities: Prototype similarities (B, T, P)
            hidden_states_aug: Augmented view hidden states (optional)
            assignments_aug: Assignments from augmented view (optional)
            mask: Boolean mask for masked positions (B, T), optional
            target_assignments: Hard assignments from unmasked view (B, T), optional

        Returns:
            total_loss: Combined loss scalar
            losses: Dictionary with all loss components and metrics
        """
        B, T, D = hidden_states.shape
        device = hidden_states.device
        losses = {}

        # Generate mask if not provided
        if mask is None:
            mask = self.mask_generator(B, T, device)

        # 1. Masked Prototype Prediction Loss
        if mask.any():
            # Use target_assignments if provided (from unmasked view),
            # otherwise fall back to hard assignments from current view
            if target_assignments is not None:
                targets = target_assignments  # (B, T)
            else:
                targets = assignments.argmax(dim=-1)  # (B, T)

            masked_loss, masked_stats = self.masked_loss(
                similarities, targets, mask
            )
            losses['masked_loss'] = masked_loss
            losses['masked_accuracy'] = torch.tensor(masked_stats['masked_accuracy'])
        else:
            losses['masked_loss'] = torch.tensor(0.0, device=device)
            losses['masked_accuracy'] = torch.tensor(0.0, device=device)

        # 2. Prototype Loss (cluster + separation)
        proto_loss, proto_losses = self.prototype_loss(
            hidden_states, prototypes, assignments
        )
        losses['prototype_loss'] = proto_loss
        losses['cluster_loss'] = proto_losses['cluster_loss']
        losses['separation_loss'] = proto_losses['separation_loss']

        # 3. Diversity Loss (counters collapse)
        div_loss, div_losses = self.diversity_loss(
            assignments, prototypes
        )
        losses['diversity_loss'] = div_loss
        losses['entropy_loss'] = div_losses['entropy_loss']
        losses['orthogonality_loss'] = div_losses['orthogonality_loss']
        losses['temporal_loss'] = div_losses['temporal_loss']
        losses['effective_prototypes'] = div_losses['effective_prototypes']
        losses['avg_proto_similarity'] = div_losses['avg_proto_similarity']

        # 4. Contrastive Loss (optional, if augmented view provided)
        if hidden_states_aug is not None:
            contrast_loss, contrast_stats = self.contrastive_loss(
                hidden_states, hidden_states_aug
            )
            losses['contrastive_loss'] = contrast_loss
            losses['contrastive_accuracy'] = torch.tensor(contrast_stats['contrastive_accuracy'])
        else:
            losses['contrastive_loss'] = torch.tensor(0.0, device=device)
            losses['contrastive_accuracy'] = torch.tensor(0.0, device=device)

        # Combined total loss
        total_loss = (
            self.config.lambda_masked * losses['masked_loss'] +
            self.config.lambda_prototype * losses['prototype_loss'] +
            self.config.lambda_diversity * losses['diversity_loss'] +
            self.config.lambda_contrastive * losses['contrastive_loss']
        )
        losses['total_loss'] = total_loss

        return total_loss, losses

    def check_collapse(
        self,
        assignments: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> Dict[str, bool]:
        """
        Check for signs of collapse.

        Returns warning flags if collapse is detected.
        """
        return self.diversity_loss.check_collapse(assignments, prototypes)

    def get_loss_weights(self) -> Dict[str, float]:
        """Return current loss weights for logging."""
        return {
            'lambda_masked': self.config.lambda_masked,
            'lambda_prototype': self.config.lambda_prototype,
            'lambda_diversity': self.config.lambda_diversity,
            'lambda_contrastive': self.config.lambda_contrastive,
        }
