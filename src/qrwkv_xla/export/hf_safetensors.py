from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from qrwkv_xla.checkpointing import CheckpointManifest, load_checkpoint
from qrwkv_xla.students import create_student

SAFETENSORS_REQUIRED_MESSAGE = (
    "safetensors is required for HF/safetensors export. "
    "Install with `pip install safetensors`."
)
EXPORT_SCHEMA_VERSION = "0.1"
CREATED_BY = "qrwkv_xla.export.hf_safetensors"
CONFIG_NAME = "config.json"
MODEL_NAME = "model.safetensors"
EXPORT_METADATA_NAME = "qrwkv_xla_export.json"
WEIGHT_MAP_NAME = "weight_map.json"


@dataclass(frozen=True)
class HfSafetensorsExport:
    export_dir: Path
    config_path: Path
    model_path: Path
    metadata_path: Path
    weight_map_path: Path
    metadata: dict[str, Any]
    weight_map: dict[str, str]


@dataclass(frozen=True)
class ExportedStudent:
    export_dir: Path
    student: Any
    params: dict[str, Any]
    config: dict[str, Any]
    metadata: dict[str, Any]
    weight_map: dict[str, str]


def export_checkpoint_to_hf_safetensors(
    checkpoint_dir: str | Path,
    export_dir: str | Path,
    *,
    overwrite: bool = False,
) -> HfSafetensorsExport:
    save_file, _ = _safetensors_numpy()
    loaded = load_checkpoint(checkpoint_dir)
    output_dir = Path(export_dir)
    _prepare_export_dir(output_dir, overwrite=overwrite)

    tensors: dict[str, np.ndarray] = {}
    weight_map: dict[str, str] = {}
    param_tree = _flatten_params_for_export(
        loaded.params,
        tensors=tensors,
        weight_map=weight_map,
        path=(),
    )
    if not tensors:
        raise ValueError("checkpoint params must contain at least one tensor")

    config = _hf_config_from_manifest(loaded.manifest)
    metadata = _export_metadata(
        checkpoint_dir=loaded.checkpoint_dir,
        manifest=loaded.manifest,
        param_tree=param_tree,
        tensor_count=len(tensors),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / CONFIG_NAME
    model_path = output_dir / MODEL_NAME
    metadata_path = output_dir / EXPORT_METADATA_NAME
    weight_map_path = output_dir / WEIGHT_MAP_NAME

    config_path.write_text(_json_dump(config), encoding="utf-8")
    save_file(tensors, model_path)
    metadata_path.write_text(_json_dump(metadata), encoding="utf-8")
    weight_map_path.write_text(_json_dump(weight_map), encoding="utf-8")

    return HfSafetensorsExport(
        export_dir=output_dir,
        config_path=config_path,
        model_path=model_path,
        metadata_path=metadata_path,
        weight_map_path=weight_map_path,
        metadata=metadata,
        weight_map=weight_map,
    )


def load_hf_safetensors_export(export_dir: str | Path) -> ExportedStudent:
    _, load_file = _safetensors_numpy()
    input_dir = Path(export_dir)
    config = _read_json(input_dir / CONFIG_NAME)
    metadata = _read_json(input_dir / EXPORT_METADATA_NAME)
    weight_map = _read_json(input_dir / WEIGHT_MAP_NAME)
    tensors = load_file(input_dir / MODEL_NAME)

    if metadata.get("schema_version") != EXPORT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported export schema_version "
            f"{metadata.get('schema_version')!r}; expected {EXPORT_SCHEMA_VERSION!r}"
        )
    if not isinstance(weight_map, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in weight_map.items()
    ):
        raise ValueError("weight_map.json must contain a string-to-string mapping")

    param_tree = metadata.get("param_tree")
    if not isinstance(param_tree, dict):
        raise ValueError("qrwkv_xla_export.json must contain param_tree")
    expected_keys: set[str] = set()
    params = _unflatten_params_from_export(param_tree, tensors, expected_keys)
    missing = expected_keys - set(tensors)
    if missing:
        raise ValueError(
            "model.safetensors is missing tensors: " + ", ".join(sorted(missing))
        )

    student_config = _student_config_from_export(config, metadata)
    student = create_student(
        str(student_config["architecture"]),
        vocab_size=int(student_config["vocab_size"]),
        hidden_size=int(student_config["hidden_size"]),
        num_layers=int(student_config["num_layers"]),
        num_heads=(
            None
            if student_config.get("num_heads") is None
            else int(student_config["num_heads"])
        ),
        num_kv_heads=(
            None
            if student_config.get("num_kv_heads") is None
            else int(student_config["num_kv_heads"])
        ),
        emit_logits=bool(student_config.get("emit_logits", False)),
        tie_embeddings=bool(student_config.get("tie_embeddings", False)),
        emit_mixer_outputs=bool(student_config.get("emit_mixer_outputs", False)),
    )
    return ExportedStudent(
        export_dir=input_dir,
        student=student,
        params=params,
        config=config,
        metadata=metadata,
        weight_map=weight_map,
    )


def _safetensors_numpy() -> tuple[Any, Any]:
    try:
        from safetensors.numpy import load_file, save_file
    except ImportError as exc:  # pragma: no cover - exercised by dependency absence.
        raise ImportError(SAFETENSORS_REQUIRED_MESSAGE) from exc
    return save_file, load_file


def _prepare_export_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not overwrite:
        existing = [
            name
            for name in (
                CONFIG_NAME,
                MODEL_NAME,
                EXPORT_METADATA_NAME,
                WEIGHT_MAP_NAME,
            )
            if (output_dir / name).exists()
        ]
        if existing:
            raise FileExistsError(
                f"export already exists at {output_dir}; pass overwrite=True"
            )


def _hf_config_from_manifest(manifest: CheckpointManifest) -> dict[str, Any]:
    student_config = dict(manifest.student_config)
    required = ("vocab_size", "hidden_size", "num_layers")
    missing = [name for name in required if name not in student_config]
    if missing:
        raise ValueError(
            "checkpoint student_config is missing required export fields: "
            + ", ".join(missing)
        )
    return {
        "model_type": "qrwkv_xla_student",
        "architectures": ["QRWKVXLAStudentForCausalLM"],
        "qrwkv_xla_architecture": manifest.student_architecture,
        "vocab_size": int(student_config["vocab_size"]),
        "hidden_size": int(student_config["hidden_size"]),
        "num_hidden_layers": int(student_config["num_layers"]),
        "num_layers": int(student_config["num_layers"]),
        "num_heads": student_config.get("num_heads"),
        "num_kv_heads": student_config.get("num_kv_heads"),
        "emit_logits": bool(student_config.get("emit_logits", False)),
        "tie_embeddings": bool(student_config.get("tie_embeddings", False)),
        "emit_mixer_outputs": bool(student_config.get("emit_mixer_outputs", False)),
        "torch_dtype": "float32",
        "transformers_version": None,
        "qrwkv_xla_export_note": (
            "Tiny QRWKV-XLA safetensors interchange config; no production "
            "Transformers model class is provided."
        ),
    }


def _export_metadata(
    *,
    checkpoint_dir: Path,
    manifest: CheckpointManifest,
    param_tree: dict[str, Any],
    tensor_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "created_by": CREATED_BY,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_checkpoint": str(checkpoint_dir),
        "checkpoint_manifest": asdict(manifest),
        "student_architecture": manifest.student_architecture,
        "student_config": dict(manifest.student_config),
        "checkpoint_step": manifest.step,
        "param_tree": param_tree,
        "tensor_count": tensor_count,
        "limitations": [
            "HF-style safetensors interchange only",
            "no production Hugging Face model class",
            "no sharding or pjit export",
            "no model quality claim",
        ],
    }


def _flatten_params_for_export(
    value: Any,
    *,
    tensors: dict[str, np.ndarray],
    weight_map: dict[str, str],
    path: tuple[str, ...],
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {
            "type": "dict",
            "children": {
                str(key): _flatten_params_for_export(
                    child,
                    tensors=tensors,
                    weight_map=weight_map,
                    path=(*path, str(key)),
                )
                for key, child in value.items()
            },
        }
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [
                _flatten_params_for_export(
                    child,
                    tensors=tensors,
                    weight_map=weight_map,
                    path=(*path, str(index)),
                )
                for index, child in enumerate(value)
            ],
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "items": [
                _flatten_params_for_export(
                    child,
                    tensors=tensors,
                    weight_map=weight_map,
                    path=(*path, str(index)),
                )
                for index, child in enumerate(value)
            ],
        }

    array = np.asarray(value)
    if array.dtype == object:
        raise TypeError(
            f"object arrays are not supported in safetensors export: {path}"
        )
    tensor_key = _tensor_key(path, len(tensors))
    tensors[tensor_key] = np.ascontiguousarray(array)
    logical_path = ".".join(path) if path else tensor_key
    weight_map[logical_path] = tensor_key
    return {
        "type": "array",
        "tensor_key": tensor_key,
        "path": list(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _unflatten_params_from_export(
    node: Any,
    tensors: Mapping[str, np.ndarray],
    expected_keys: set[str],
) -> Any:
    if not isinstance(node, dict):
        raise ValueError("export param_tree nodes must be mappings")
    node_type = node.get("type")
    if node_type == "dict":
        children = node.get("children")
        if not isinstance(children, dict):
            raise ValueError("dict export param_tree node must contain children")
        return {
            key: _unflatten_params_from_export(child, tensors, expected_keys)
            for key, child in children.items()
        }
    if node_type in {"list", "tuple"}:
        items = node.get("items")
        if not isinstance(items, list):
            raise ValueError(f"{node_type} export param_tree node must contain items")
        values = [
            _unflatten_params_from_export(child, tensors, expected_keys)
            for child in items
        ]
        return tuple(values) if node_type == "tuple" else values
    if node_type == "array":
        tensor_key = _required_str(node, "tensor_key")
        expected_keys.add(tensor_key)
        if tensor_key not in tensors:
            raise ValueError(f"model.safetensors is missing tensor: {tensor_key}")
        array = np.asarray(tensors[tensor_key])
        expected_shape = tuple(_required_list(node, "shape"))
        if array.shape != expected_shape:
            raise ValueError(
                f"export tensor {tensor_key} has shape {array.shape}, "
                f"expected {expected_shape}"
            )
        expected_dtype = _required_str(node, "dtype")
        if str(array.dtype) != expected_dtype:
            raise ValueError(
                f"export tensor {tensor_key} has dtype {array.dtype}, "
                f"expected {expected_dtype}"
            )
        return array
    raise ValueError(f"unknown export param_tree node type: {node_type!r}")


def _student_config_from_export(
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    raw = metadata.get("student_config")
    if not isinstance(raw, dict):
        raise ValueError("qrwkv_xla_export.json must contain student_config")
    student_config = dict(raw)
    student_config.setdefault("architecture", metadata.get("student_architecture"))
    student_config.setdefault("vocab_size", config.get("vocab_size"))
    student_config.setdefault("hidden_size", config.get("hidden_size"))
    student_config.setdefault(
        "num_layers",
        config.get("num_layers", config.get("num_hidden_layers")),
    )
    student_config.setdefault("num_heads", config.get("num_heads"))
    student_config.setdefault("num_kv_heads", config.get("num_kv_heads"))
    required = ("architecture", "vocab_size", "hidden_size", "num_layers")
    missing = [name for name in required if student_config.get(name) is None]
    if missing:
        raise ValueError(
            "export metadata is missing required student fields: " + ", ".join(missing)
        )
    return student_config


def _tensor_key(path: tuple[str, ...], index: int) -> str:
    if not path:
        return f"params.tensor_{index:06d}"
    clean = [part.replace("/", "_").replace(" ", "_") for part in path]
    return "params." + ".".join(clean)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing export file: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return raw


def _json_dump(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _required_str(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"export param_tree {key} must be a non-empty string")
    return value


def _required_list(raw: Mapping[str, Any], key: str) -> list[Any]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise ValueError(f"export param_tree {key} must be a list")
    return value
