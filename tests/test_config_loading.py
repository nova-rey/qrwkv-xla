from __future__ import annotations

from pathlib import Path

import pytest

from qrwkv_xla.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_tiny_cpu_config_loads() -> None:
    config = load_config(ROOT / "configs" / "tiny_cpu.yaml")
    assert config.runtime.backend == "cpu"
    assert config.model.hidden_size == 128
    assert config.training.batch_size == 2


def test_missing_sections_use_defaults(tmp_path: Path) -> None:
    path = tmp_path / "partial.yaml"
    path.write_text("runtime:\n  backend: cpu\n", encoding="utf-8")
    config = load_config(path)
    assert config.model.sequence_length == 64
    assert config.training.max_steps == 10


@pytest.mark.parametrize("backend", ["metal", "", "neon"])
def test_invalid_backend_raises(backend: str, tmp_path: Path) -> None:
    path = tmp_path / "invalid_backend.yaml"
    path.write_text(f"runtime:\n  backend: {backend!r}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime.backend"):
        load_config(path)


@pytest.mark.parametrize(
    ("field_name", "yaml_body"),
    [
        ("hidden_size", "model:\n  hidden_size: 0\n"),
        ("num_layers", "model:\n  num_layers: -1\n"),
        ("sequence_length", "model:\n  sequence_length: 0\n"),
        ("batch_size", "training:\n  batch_size: 0\n"),
        ("max_steps", "training:\n  max_steps: -5\n"),
    ],
)
def test_invalid_dimensions_raise(
    field_name: str, yaml_body: str, tmp_path: Path
) -> None:
    path = tmp_path / f"invalid_{field_name}.yaml"
    path.write_text(yaml_body, encoding="utf-8")
    with pytest.raises(ValueError, match=field_name):
        load_config(path)
