from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


class AttentionCaptureError(RuntimeError):
    pass


def attention_outputs_from_model_output(
    output: Any,
    *,
    torch: Any,
) -> np.ndarray:
    """Extract manually supplied HF attention-output vectors from model output."""
    candidates = (
        getattr(output, "attention_targets", None),
        getattr(output, "attention_outputs", None),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        return stack_attention_output_sequence(candidate, torch=torch)
    raise AttentionCaptureError(
        "HF model output did not expose attention_outputs or attention_targets. "
        "Enable attention_capture or use a manual model wrapper for attention "
        "target export."
    )


def capture_module_outputs(
    model: Any,
    *,
    module_names: Sequence[str],
    forward_fn: Callable[[], Any],
    torch: Any,
    output_index: int = 0,
) -> np.ndarray:
    if not module_names:
        raise AttentionCaptureError("module_names must be non-empty")
    if not hasattr(model, "named_modules"):
        raise AttentionCaptureError("model does not expose named_modules()")

    modules = {name: module for name, module in model.named_modules()}
    missing = [name for name in module_names if name not in modules]
    if missing:
        raise AttentionCaptureError(
            f"attention capture modules not found: {', '.join(missing)}"
        )

    captured: dict[str, Any] = {}
    handles = []

    def make_hook(name: str) -> Callable[[Any, Any, Any], None]:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            captured[name] = normalize_attention_output(
                output, output_index=output_index
            )

        return hook

    for name in module_names:
        module = modules[name]
        if not hasattr(module, "register_forward_hook"):
            raise AttentionCaptureError(
                f"module {name!r} does not support forward hooks"
            )
        handles.append(module.register_forward_hook(make_hook(name)))

    try:
        forward_fn()
    finally:
        for handle in handles:
            handle.remove()

    if len(captured) != len(module_names):
        missing_after = [name for name in module_names if name not in captured]
        raise AttentionCaptureError(
            "attention capture did not observe all requested modules: "
            + ", ".join(missing_after)
        )

    ordered = [captured[name] for name in module_names]
    return stack_attention_output_sequence(ordered, torch=torch)


def discover_auto_qwen_module_names(model: Any) -> tuple[str, ...]:
    if not hasattr(model, "named_modules"):
        raise AttentionCaptureError("model does not expose named_modules()")

    matches: list[tuple[int, str]] = []
    for name, _module in model.named_modules():
        if ".self_attn" not in name:
            continue
        layer_index = _extract_layer_index(name)
        if layer_index is None:
            continue
        matches.append((layer_index, name))
    if not matches:
        raise AttentionCaptureError(
            "auto_qwen attention capture could not find any *.self_attn modules"
        )
    matches.sort()
    return tuple(name for _index, name in matches)


def stack_attention_output_sequence(values: Sequence[Any], *, torch: Any) -> np.ndarray:
    if not values:
        raise AttentionCaptureError("attention output sequence is empty")
    stacked = torch.stack(tuple(values), dim=1)
    array = _tensor_to_numpy(stacked).astype(np.float32)
    validate_attention_output_array(array)
    return array


def validate_attention_output_array(array: np.ndarray) -> None:
    if array.ndim != 4:
        raise AttentionCaptureError(
            f"attention output vectors must have rank 4, got shape {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise AttentionCaptureError("attention output vectors must be floating dtype")


def normalize_attention_output(output: Any, *, output_index: int) -> Any:
    if isinstance(output, (tuple, list)):
        if output_index >= len(output):
            raise AttentionCaptureError(
                f"attention capture output_index {output_index} is out of range"
            )
        output = output[output_index]
    array = _tensor_to_numpy(output)
    if array.ndim != 3:
        raise AttentionCaptureError(
            "attention module outputs must have shape [batch, sequence_length, "
            "hidden_size], "
            f"got {array.shape}"
        )
    return output


def _extract_layer_index(module_name: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", module_name)
    if match is None:
        return None
    return int(match.group(1))


def _tensor_to_numpy(tensor: Any) -> np.ndarray:
    value = tensor
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)
