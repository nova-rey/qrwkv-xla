from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from qrwkv_xla.artifacts import FingerprintBatch
from qrwkv_xla.training.fingerprint_stats import FingerprintDistributionStats


@dataclass(frozen=True)
class FingerprintCorridorLossConfig:
    entropy_weight: float = 1.0
    top1_margin_weight: float = 1.0
    top8_mass_weight: float = 1.0
    top32_mass_weight: float = 1.0
    tail_mass_weight: float = 1.0
    use_record_weights: bool = True
    eps: float = 1e-8


@dataclass(frozen=True)
class FingerprintCorridorLossOutput:
    loss: jax.Array
    entropy_loss: jax.Array
    top1_margin_loss: jax.Array
    top8_mass_loss: jax.Array
    top32_mass_loss: jax.Array
    tail_mass_loss: jax.Array
    entropy_inside_rate: jax.Array
    top1_margin_inside_rate: jax.Array
    top8_mass_inside_rate: jax.Array
    top32_mass_inside_rate: jax.Array
    tail_mass_inside_rate: jax.Array
    all_inside_rate: jax.Array
    mean_weight: jax.Array


def squared_hinge_bound_penalty(
    values: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
) -> jax.Array:
    values = jnp.asarray(values)
    lower = jnp.asarray(lower)
    upper = jnp.asarray(upper)
    _check_same_rank1_shape(
        ("values", values),
        ("lower", lower),
        ("upper", upper),
    )
    below = jnp.maximum(lower - values, 0.0)
    above = jnp.maximum(values - upper, 0.0)
    return jnp.square(below) + jnp.square(above)


def inside_bounds(
    values: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
) -> jax.Array:
    values = jnp.asarray(values)
    lower = jnp.asarray(lower)
    upper = jnp.asarray(upper)
    _check_same_rank1_shape(
        ("values", values),
        ("lower", lower),
        ("upper", upper),
    )
    return (values >= lower) & (values <= upper)


def compute_fingerprint_corridor_loss(
    stats: FingerprintDistributionStats,
    batch: FingerprintBatch,
    config: FingerprintCorridorLossConfig | None = None,
) -> FingerprintCorridorLossOutput:
    cfg = config or FingerprintCorridorLossConfig()
    _validate_config(cfg)

    arrays = _arrays(stats, batch)
    batch_size = _validate_loss_shapes(arrays)
    weights = (
        jnp.asarray(batch.weight, dtype=jnp.float32)
        if cfg.use_record_weights
        else jnp.ones((batch_size,), dtype=jnp.float32)
    )
    if weights.ndim != 1 or weights.shape[0] != batch_size:
        raise ValueError(
            "batch.weight must be rank 1 with batch dimension "
            f"{batch_size}, got {weights.shape}"
        )

    entropy_penalty = squared_hinge_bound_penalty(
        arrays["entropy"],
        arrays["entropy_min"],
        arrays["entropy_max"],
    )
    top1_margin_penalty = squared_hinge_bound_penalty(
        arrays["top1_margin"],
        arrays["top1_margin_min"],
        arrays["top1_margin_max"],
    )
    top8_mass_penalty = squared_hinge_bound_penalty(
        arrays["top8_mass"],
        arrays["top8_mass_min"],
        arrays["top8_mass_max"],
    )
    top32_mass_penalty = squared_hinge_bound_penalty(
        arrays["top32_mass"],
        arrays["top32_mass_min"],
        arrays["top32_mass_max"],
    )
    tail_mass_penalty = squared_hinge_bound_penalty(
        arrays["tail_mass"],
        arrays["tail_mass_min"],
        arrays["tail_mass_max"],
    )

    entropy_loss = _weighted_mean(
        entropy_penalty * cfg.entropy_weight,
        weights,
        cfg.eps,
    )
    top1_margin_loss = _weighted_mean(
        top1_margin_penalty * cfg.top1_margin_weight,
        weights,
        cfg.eps,
    )
    top8_mass_loss = _weighted_mean(
        top8_mass_penalty * cfg.top8_mass_weight,
        weights,
        cfg.eps,
    )
    top32_mass_loss = _weighted_mean(
        top32_mass_penalty * cfg.top32_mass_weight,
        weights,
        cfg.eps,
    )
    tail_mass_loss = _weighted_mean(
        tail_mass_penalty * cfg.tail_mass_weight,
        weights,
        cfg.eps,
    )
    loss = (
        entropy_loss
        + top1_margin_loss
        + top8_mass_loss
        + top32_mass_loss
        + tail_mass_loss
    )

    entropy_inside = inside_bounds(
        arrays["entropy"],
        arrays["entropy_min"],
        arrays["entropy_max"],
    )
    top1_margin_inside = inside_bounds(
        arrays["top1_margin"],
        arrays["top1_margin_min"],
        arrays["top1_margin_max"],
    )
    top8_mass_inside = inside_bounds(
        arrays["top8_mass"],
        arrays["top8_mass_min"],
        arrays["top8_mass_max"],
    )
    top32_mass_inside = inside_bounds(
        arrays["top32_mass"],
        arrays["top32_mass_min"],
        arrays["top32_mass_max"],
    )
    tail_mass_inside = inside_bounds(
        arrays["tail_mass"],
        arrays["tail_mass_min"],
        arrays["tail_mass_max"],
    )
    all_inside = (
        entropy_inside
        & top1_margin_inside
        & top8_mass_inside
        & top32_mass_inside
        & tail_mass_inside
    )

    return FingerprintCorridorLossOutput(
        loss=loss,
        entropy_loss=entropy_loss,
        top1_margin_loss=top1_margin_loss,
        top8_mass_loss=top8_mass_loss,
        top32_mass_loss=top32_mass_loss,
        tail_mass_loss=tail_mass_loss,
        entropy_inside_rate=_inside_rate(entropy_inside),
        top1_margin_inside_rate=_inside_rate(top1_margin_inside),
        top8_mass_inside_rate=_inside_rate(top8_mass_inside),
        top32_mass_inside_rate=_inside_rate(top32_mass_inside),
        tail_mass_inside_rate=_inside_rate(tail_mass_inside),
        all_inside_rate=_inside_rate(all_inside),
        mean_weight=jnp.mean(weights),
    )


def _arrays(
    stats: FingerprintDistributionStats,
    batch: FingerprintBatch,
) -> dict[str, jax.Array]:
    return {
        "entropy": jnp.asarray(stats.entropy),
        "top1_margin": jnp.asarray(stats.top1_margin),
        "top8_mass": jnp.asarray(stats.top8_mass),
        "top32_mass": jnp.asarray(stats.top32_mass),
        "tail_mass": jnp.asarray(stats.tail_mass),
        "entropy_min": jnp.asarray(batch.entropy_min),
        "entropy_max": jnp.asarray(batch.entropy_max),
        "top1_margin_min": jnp.asarray(batch.top1_margin_min),
        "top1_margin_max": jnp.asarray(batch.top1_margin_max),
        "top8_mass_min": jnp.asarray(batch.top8_mass_min),
        "top8_mass_max": jnp.asarray(batch.top8_mass_max),
        "top32_mass_min": jnp.asarray(batch.top32_mass_min),
        "top32_mass_max": jnp.asarray(batch.top32_mass_max),
        "tail_mass_min": jnp.asarray(batch.tail_mass_min),
        "tail_mass_max": jnp.asarray(batch.tail_mass_max),
    }


def _validate_loss_shapes(arrays: dict[str, jax.Array]) -> int:
    batch_size: int | None = None
    for name, array in arrays.items():
        if array.ndim != 1:
            raise ValueError(f"{name} must be rank 1 [batch], got {array.shape}")
        if batch_size is None:
            batch_size = array.shape[0]
        elif array.shape[0] != batch_size:
            raise ValueError(
                f"{name} batch dimension mismatch: expected {batch_size}, "
                f"got {array.shape[0]}"
            )
    return int(batch_size or 0)


def _check_same_rank1_shape(*items: tuple[str, jax.Array]) -> None:
    expected_shape = None
    for name, array in items:
        if array.ndim != 1:
            raise ValueError(f"{name} must be rank 1 [batch], got {array.shape}")
        if expected_shape is None:
            expected_shape = array.shape
        elif array.shape != expected_shape:
            raise ValueError(
                f"{name} shape mismatch: expected {expected_shape}, got {array.shape}"
            )


def _validate_config(config: FingerprintCorridorLossConfig) -> None:
    weights = (
        config.entropy_weight,
        config.top1_margin_weight,
        config.top8_mass_weight,
        config.top32_mass_weight,
        config.tail_mass_weight,
    )
    if any(weight < 0 for weight in weights):
        raise ValueError("fingerprint corridor loss weights must be non-negative")
    if config.eps <= 0:
        raise ValueError(f"eps must be positive, got {config.eps}")


def _weighted_mean(values: jax.Array, weights: jax.Array, eps: float) -> jax.Array:
    normalizer = jnp.maximum(jnp.sum(weights), jnp.asarray(eps, dtype=weights.dtype))
    return jnp.sum(values * weights) / normalizer


def _inside_rate(mask: jax.Array) -> jax.Array:
    return jnp.mean(mask.astype(jnp.float32))
