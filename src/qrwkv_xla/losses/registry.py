from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from qrwkv_xla.losses.hidden import hidden_mse_loss
from qrwkv_xla.losses.logits import logits_kl_loss

LossFn = Callable[..., Any]


@dataclass(frozen=True)
class LossTerm:
    name: str
    weight: float
    options: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError(f"loss weight must be >= 0 for {self.name!r}")


@dataclass(frozen=True)
class WeightedLoss:
    total: jax.Array
    components: dict[str, jax.Array]


_REGISTRY: dict[str, LossFn] = {
    "hidden_mse": hidden_mse_loss,
    "logits_kl": logits_kl_loss,
}


def registered_loss_names() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_loss(name: str) -> LossFn:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        allowed = ", ".join(registered_loss_names())
        raise ValueError(
            f"Unknown distillation loss {name!r}; expected one of {allowed}"
        ) from exc


def compose_weighted_loss(terms: tuple[LossTerm, ...]) -> Callable[..., WeightedLoss]:
    active_terms = tuple(term for term in terms if term.weight > 0.0)
    if not active_terms:
        raise ValueError("At least one enabled distillation loss must have weight > 0")
    for term in active_terms:
        get_loss(term.name)

    def loss_fn(student_output: Any, batch: Mapping[str, Any]) -> WeightedLoss:
        components: dict[str, jax.Array] = {}
        total: jax.Array | None = None
        for term in active_terms:
            raw_loss = _evaluate_term(term, student_output, batch)
            weighted = raw_loss * jnp.asarray(term.weight, dtype=raw_loss.dtype)
            components[term.name] = raw_loss
            components[f"{term.name}_weighted"] = weighted
            total = weighted if total is None else total + weighted
        if total is None:
            total = jnp.asarray(0.0)
        components["loss"] = total
        return WeightedLoss(total=total, components=components)

    return loss_fn


def _evaluate_term(
    term: LossTerm, student_output: Any, batch: Mapping[str, Any]
) -> jax.Array:
    options = dict(term.options or {})
    if term.name == "hidden_mse":
        return get_loss(term.name)(
            student_output.hidden_states,
            batch["hidden_states"],
            batch.get("attention_mask"),
            **options,
        )
    if term.name == "logits_kl":
        if student_output.logits is None:
            raise ValueError("logits_kl is enabled but the student did not emit logits")
        if "logits" not in batch:
            raise ValueError("logits_kl is enabled but the target batch has no logits")
        return get_loss(term.name)(
            student_output.logits,
            batch["logits"],
            batch.get("attention_mask"),
            **options,
        )
    raise ValueError(f"Unsupported distillation loss term: {term.name}")
