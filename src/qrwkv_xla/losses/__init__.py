"""Losses for QRWKV-XLA."""

from qrwkv_xla.losses.hidden import hidden_mse_loss
from qrwkv_xla.losses.logits import logits_kl_loss
from qrwkv_xla.losses.registry import (
    LossTerm,
    WeightedLoss,
    compose_weighted_loss,
    get_loss,
    registered_loss_names,
)

__all__ = [
    "LossTerm",
    "WeightedLoss",
    "compose_weighted_loss",
    "get_loss",
    "hidden_mse_loss",
    "logits_kl_loss",
    "registered_loss_names",
]
