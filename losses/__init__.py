"""ProtoSpeech Losses."""

from .diversity_loss import DiversityLoss, EntropyLoss, OrthogonalityLoss
from .prototype_loss import PrototypeLoss, ClusterLoss, SeparationLoss
from .masked_prediction_loss import MaskedPrototypePredictionLoss
from .contrastive_loss import ContrastiveLoss
from .combined_loss import ProtoSpeechLoss

# Temporal diversity losses (ProtoSpeech-TD)
from .temporal_losses import (
    TemporalTransitionLoss,
    CrossTimeContrastiveLoss,
    PrototypeDiversityLoss,
    TemporalDiversityLoss,
)

__all__ = [
    'DiversityLoss',
    'EntropyLoss',
    'OrthogonalityLoss',
    'PrototypeLoss',
    'ClusterLoss',
    'SeparationLoss',
    'MaskedPrototypePredictionLoss',
    'ContrastiveLoss',
    'ProtoSpeechLoss',
    # Temporal diversity losses
    'TemporalTransitionLoss',
    'CrossTimeContrastiveLoss',
    'PrototypeDiversityLoss',
    'TemporalDiversityLoss',
]
