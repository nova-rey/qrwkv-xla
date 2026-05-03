from __future__ import annotations

import numpy as np
import pytest

from qrwkv_xla.teacher_export.attention_capture import (
    AttentionCaptureError,
    capture_module_outputs,
    discover_auto_qwen_module_names,
    normalize_attention_output,
)


class FakeTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = np.asarray(value)

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class FakeHookHandle:
    def __init__(self, hooks: list, hook) -> None:
        self._hooks = hooks
        self._hook = hook

    def remove(self) -> None:
        self._hooks.remove(self._hook)


class FakeModule:
    def __init__(self, output) -> None:
        self.output = output
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return FakeHookHandle(self.hooks, hook)

    def forward(self):
        for hook in list(self.hooks):
            hook(self, (), self.output)


class FakeModel:
    def __init__(self, modules: dict[str, FakeModule]) -> None:
        self.modules = modules

    def named_modules(self):
        yield "", self
        yield from self.modules.items()


class FakeTorch:
    @staticmethod
    def stack(values, dim: int):
        return FakeTensor(np.stack([value.value for value in values], axis=dim))


def test_normalize_attention_output_accepts_tuple_index_zero() -> None:
    tensor = FakeTensor(np.ones((1, 2, 3), dtype=np.float32))
    result = normalize_attention_output((tensor, "ignored"), output_index=0)
    assert result is tensor


def test_capture_module_outputs_stacks_requested_modules() -> None:
    first = FakeModule(FakeTensor(np.ones((1, 2, 3), dtype=np.float32)))
    second = FakeModule((FakeTensor(np.full((1, 2, 3), 2.0, dtype=np.float32)),))
    model = FakeModel(
        {
            "model.layers.0.self_attn": first,
            "model.layers.1.self_attn": second,
        }
    )

    captured = capture_module_outputs(
        model,
        module_names=("model.layers.0.self_attn", "model.layers.1.self_attn"),
        forward_fn=lambda: [module.forward() for module in model.modules.values()],
        torch=FakeTorch,
        output_index=0,
    )

    assert captured.shape == (1, 2, 2, 3)
    assert np.allclose(captured[:, 0], 1.0)
    assert np.allclose(captured[:, 1], 2.0)


def test_discover_auto_qwen_module_names_orders_layers() -> None:
    model = FakeModel(
        {
            "model.layers.2.self_attn": FakeModule(None),
            "model.layers.0.self_attn": FakeModule(None),
            "model.layers.1.self_attn": FakeModule(None),
        }
    )
    names = discover_auto_qwen_module_names(model)
    assert names == (
        "model.layers.0.self_attn",
        "model.layers.1.self_attn",
        "model.layers.2.self_attn",
    )


def test_capture_module_outputs_raises_for_missing_module() -> None:
    model = FakeModel({"model.layers.0.self_attn": FakeModule(None)})
    with pytest.raises(AttentionCaptureError, match="not found"):
        capture_module_outputs(
            model,
            module_names=("model.layers.1.self_attn",),
            forward_fn=lambda: None,
            torch=FakeTorch,
        )
