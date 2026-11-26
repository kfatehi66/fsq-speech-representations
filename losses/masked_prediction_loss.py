"""
Masked Prototype Prediction Loss

Instead of predicting k-means cluster IDs (like HuBERT), we predict
prototype assignments. This enables the model to learn meaningful
prototypes in a self-supervised manner.
"""

from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedPrototypePredictionLoss(nn.Module):
    """
    Masked Prototype Prediction Loss.

    Given masked positions, predict the prototype assignment that would
    have been made without masking. This is similar to HuBERT's masked
    cluster prediction but with learnable prototypes.

    L_masked = -1/|M| * Σ_{i∈M} Σ_j a*_ij * log(â_ij)

    Where:
    - M = set of masked positions
    - a*_ij = target assignment (from unmasked/teacher view)
    - â_ij = predicted assignment (from masked/student view)

    Args:
        temperature: Temperature for target distribution (default: 0.1)
        label_smoothing: Label smoothing factor (default: 0.0)
    """

    def __init__(
        self,
        temperature: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute masked prototype prediction loss.

        Args:
            predictions: Predicted similarities/logits (B, T, P) from masked view
            targets: Target assignments (B, T, P) from unmasked view (soft)
                    or target indices (B, T) from unmasked view (hard)
            mask: Boolean mask (B, T) where True = masked position

        Returns:
            loss: Scalar masked prediction loss
            stats: Dictionary with accuracy and loss statistics
        """
        B, T, P = predictions.shape

        # Get masked positions
        masked_preds = predictions[mask]  # (num_masked, P)

        if masked_preds.numel() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True), {
                'masked_loss': 0.0,
                'masked_accuracy': 0.0,
                'num_masked': 0,
            }

        # Handle both soft and hard targets
        if targets.dim() == 2:
            # Hard targets: (B, T) indices
            masked_targets = targets[mask]  # (num_masked,)

            # Apply label smoothing if specified
            if self.label_smoothing > 0:
                # Convert to soft targets with smoothing
                num_masked = masked_targets.size(0)
                soft_targets = torch.zeros(num_masked, P, device=targets.device)
                soft_targets.scatter_(1, masked_targets.unsqueeze(1), 1.0)
                soft_targets = soft_targets * (1 - self.label_smoothing) + self.label_smoothing / P
                masked_targets = soft_targets

                # KL divergence loss
                log_preds = F.log_softmax(masked_preds / self.temperature, dim=-1)
                loss = F.kl_div(log_preds, masked_targets, reduction='batchmean')
            else:
                # Cross-entropy loss
                loss = F.cross_entropy(
                    masked_preds / self.temperature,
                    masked_targets,
                )
            # Compute accuracy
            pred_indices = masked_preds.argmax(dim=-1)
            if targets.dim() == 2:
                target_indices = targets[mask]
            else:
                target_indices = targets[mask].argmax(dim=-1)
            accuracy = (pred_indices == target_indices).float().mean().item()

        else:
            # Soft targets: (B, T, P) probabilities
            masked_targets = targets[mask]  # (num_masked, P)

            # Sharpen targets with temperature
            sharpened_targets = F.softmax(
                torch.log(masked_targets + 1e-10) / self.temperature,
                dim=-1
            )

            # KL divergence loss
            log_preds = F.log_softmax(masked_preds / self.temperature, dim=-1)
            loss = F.kl_div(log_preds, sharpened_targets, reduction='batchmean')

            # Compute accuracy (using argmax)
            pred_indices = masked_preds.argmax(dim=-1)
            target_indices = masked_targets.argmax(dim=-1)
            accuracy = (pred_indices == target_indices).float().mean().item()

        stats = {
            'masked_loss': loss.item(),
            'masked_accuracy': accuracy,
            'num_masked': mask.sum().item(),
        }

        return loss, stats


class SpanMaskGenerator(nn.Module):
    """
    Generate span masks for masked prediction.

    Similar to wav2vec2/HuBERT masking strategy:
    - Select starting positions randomly
    - Extend each position to a span of fixed length
    - Target ~15% of positions masked

    Args:
        mask_prob: Probability of masking (default: 0.15)
        mask_span_length: Average span length (default: 10)
        min_masks: Minimum number of mask spans (default: 1)
    """

    def __init__(
        self,
        mask_prob: float = 0.15,
        mask_span_length: int = 10,
        min_masks: int = 1,
    ):
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_span_length = mask_span_length
        self.min_masks = min_masks

    def forward(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate span mask.

        Args:
            batch_size: Number of samples in batch
            seq_len: Sequence length
            device: Device for tensor

        Returns:
            mask: Boolean mask (B, T) where True = masked
        """
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        # Number of frames to mask per sample
        num_to_mask = int(seq_len * self.mask_prob)
        num_spans = max(self.min_masks, num_to_mask // self.mask_span_length)

        for b in range(batch_size):
            # Random span starts
            max_start = max(1, seq_len - self.mask_span_length)
            starts = torch.randint(0, max_start, (num_spans,), device=device)

            for start in starts:
                end = min(start + self.mask_span_length, seq_len)
                mask[b, start:end] = True

        return mask

    def get_mask_indices(
        self,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert boolean mask to indices.

        Args:
            mask: Boolean mask (B, T)

        Returns:
            indices: Flat indices of masked positions
        """
        return mask.nonzero(as_tuple=False)
