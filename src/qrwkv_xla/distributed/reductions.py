from __future__ import annotations

from typing import Any

import jax


def tree_pmean(tree: Any, *, axis_name: str) -> Any:
    return jax.tree_util.tree_map(lambda leaf: jax.lax.pmean(leaf, axis_name), tree)


def metrics_pmean(metrics: dict[str, Any], *, axis_name: str) -> dict[str, Any]:
    return {name: jax.lax.pmean(value, axis_name) for name, value in metrics.items()}
