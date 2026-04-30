"""Training module for CBF-based control."""

from .config import TrainingConfig
from .trainer import CBFTrainer

__all__ = [
    'TrainingConfig',
    'CBFTrainer',
]
