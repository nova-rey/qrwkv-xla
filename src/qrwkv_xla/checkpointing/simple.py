from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qrwkv_xla.optimizers import OptimizerState

SCHEMA_VERSION = "0.1"
CREATED_BY = "qrwkv_xla.checkpointing.simple"
MANIFEST_NAME = "checkpoint.json"
PARAMS_NAME = "params.npz"


@dataclass(frozen=True)
class CheckpointManifest:
    schema_version: str
    created_by: str
    student_architecture: str
    student_config: dict[str, Any]
    step: int
    learning_rate: float
    loss_config: dict[str, Any]
    target_manifest: dict[str, Any]
    param_tree: dict[str, Any]
    optimizer_config: dict[str, Any] = field(default_factory=dict)
    optimizer_state: dict[str, Any] | None = None
    lr_schedule: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LoadedCheckpoint:
    checkpoint_dir: Path
    manifest: CheckpointManifest
    params: Any
    optimizer_state: OptimizerState | None = None


def checkpoint_exists(checkpoint_dir: str | Path) -> bool:
    checkpoint_path = Path(checkpoint_dir)
    return (checkpoint_path / MANIFEST_NAME).is_file() and (
        checkpoint_path / PARAMS_NAME
    ).is_file()


def save_checkpoint(
    checkpoint_dir: str | Path,
    params: Any,
    *,
    student_architecture: str,
    student_config: Mapping[str, Any],
    step: int,
    learning_rate: float,
    loss_config: Any,
    target_manifest: Any,
    optimizer_config: Any | None = None,
    optimizer_state: OptimizerState | None = None,
    lr_schedule: Any | None = None,
    notes: Sequence[str] | None = None,
    overwrite: bool = False,
    created_by: str = CREATED_BY,
) -> Path:
    output_dir = Path(checkpoint_dir)
    _validate_checkpoint_path(output_dir)
    _validate_save_request(
        output_dir=output_dir,
        params=params,
        student_architecture=student_architecture,
        student_config=student_config,
        step=step,
        learning_rate=learning_rate,
        overwrite=overwrite,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    param_tree = _flatten_tree(params, arrays, ())
    if not arrays:
        raise ValueError("params must contain at least one array leaf")
    optimizer_state_tree = None
    if optimizer_state is not None:
        optimizer_state_tree = {
            "type": optimizer_state.type,
            "step": _scalar_int(optimizer_state.step),
            "slots_tree": _flatten_tree(
                optimizer_state.slots,
                arrays,
                ("optimizer", "slots"),
            ),
        }
    np.savez(output_dir / PARAMS_NAME, **arrays)

    manifest = CheckpointManifest(
        schema_version=SCHEMA_VERSION,
        created_by=created_by,
        student_architecture=student_architecture,
        student_config=dict(student_config),
        step=int(step),
        learning_rate=float(learning_rate),
        loss_config=_required_json_dict(_jsonable(loss_config), "loss_config"),
        target_manifest=_required_json_dict(
            _jsonable(target_manifest),
            "target_manifest",
        ),
        param_tree=param_tree,
        optimizer_config=_required_json_dict(
            _jsonable(optimizer_config or {}),
            "optimizer_config",
        ),
        optimizer_state=optimizer_state_tree,
        lr_schedule=_required_json_dict(_jsonable(lr_schedule or {}), "lr_schedule"),
        notes=list(notes or []),
    )
    validate_checkpoint_manifest(manifest)
    (output_dir / MANIFEST_NAME).write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_dir


def load_checkpoint(checkpoint_dir: str | Path) -> LoadedCheckpoint:
    input_dir = Path(checkpoint_dir)
    manifest_path = input_dir / MANIFEST_NAME
    params_path = input_dir / PARAMS_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing checkpoint manifest: {manifest_path}")
    if not params_path.is_file():
        raise FileNotFoundError(f"missing checkpoint params archive: {params_path}")

    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _parse_manifest(raw_manifest)
    with np.load(params_path, allow_pickle=False) as archive:
        available_keys = set(archive.files)
        expected_keys: set[str] = set()
        params = _unflatten_tree(manifest.param_tree, archive, expected_keys)
        optimizer_state = _unflatten_optimizer_state(
            manifest.optimizer_state,
            archive,
            expected_keys,
        )
        missing = expected_keys - available_keys
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"checkpoint params.npz is missing arrays: {missing_text}")
    return LoadedCheckpoint(
        checkpoint_dir=input_dir,
        manifest=manifest,
        params=params,
        optimizer_state=optimizer_state,
    )


def validate_checkpoint_manifest(manifest: CheckpointManifest) -> None:
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError(
            "unsupported checkpoint schema_version "
            f"{manifest.schema_version!r}; expected {SCHEMA_VERSION!r}"
        )
    if not manifest.created_by:
        raise ValueError("checkpoint manifest created_by must be non-empty")
    if not manifest.student_architecture:
        raise ValueError("checkpoint manifest student_architecture must be non-empty")
    if not manifest.student_config:
        raise ValueError("checkpoint manifest student_config must be non-empty")
    if manifest.step < 0:
        raise ValueError("checkpoint manifest step must be >= 0")
    if manifest.learning_rate <= 0:
        raise ValueError("checkpoint manifest learning_rate must be > 0")
    if not isinstance(manifest.loss_config, dict):
        raise ValueError("checkpoint manifest loss_config must be a mapping")
    if not isinstance(manifest.target_manifest, dict):
        raise ValueError("checkpoint manifest target_manifest must be a mapping")
    if not isinstance(manifest.param_tree, dict):
        raise ValueError("checkpoint manifest param_tree must be a mapping")
    if not isinstance(manifest.optimizer_config, dict):
        raise ValueError("checkpoint manifest optimizer_config must be a mapping")
    if not isinstance(manifest.lr_schedule, dict):
        raise ValueError("checkpoint manifest lr_schedule must be a mapping")
    if manifest.optimizer_state is not None:
        if not isinstance(manifest.optimizer_state, dict):
            raise ValueError("checkpoint manifest optimizer_state must be a mapping")
        if manifest.optimizer_state.get("type") not in {"sgd", "adam", "adamw"}:
            raise ValueError("checkpoint manifest optimizer_state type is invalid")
        if int(manifest.optimizer_state.get("step", -1)) < 0:
            raise ValueError("checkpoint manifest optimizer_state step must be >= 0")
        if not isinstance(manifest.optimizer_state.get("slots_tree"), dict):
            raise ValueError(
                "checkpoint manifest optimizer_state slots_tree must be a mapping"
            )
    if not isinstance(manifest.notes, list) or not all(
        isinstance(note, str) for note in manifest.notes
    ):
        raise ValueError("checkpoint manifest notes must be a list of strings")


def _validate_checkpoint_path(path: Path) -> None:
    parts = path.parts
    if "checkpoints" not in parts:
        raise ValueError("checkpoint paths must live under a checkpoints/ directory")


def _validate_save_request(
    *,
    output_dir: Path,
    params: Any,
    student_architecture: str,
    student_config: Mapping[str, Any],
    step: int,
    learning_rate: float,
    overwrite: bool,
) -> None:
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"checkpoint already exists at {output_dir}; pass overwrite=True"
        )
    if step < 0:
        raise ValueError("checkpoint step must be >= 0")
    if not student_architecture:
        raise ValueError("student_architecture must be non-empty")
    if not student_config:
        raise ValueError("student_config must be non-empty")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if _is_empty_tree(params):
        raise ValueError("params must be non-empty")


def _is_empty_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return not value or all(_is_empty_tree(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return not value or all(_is_empty_tree(child) for child in value)
    return False


def _flatten_tree(value: Any, arrays: dict[str, np.ndarray], path: tuple[str, ...]):
    if isinstance(value, Mapping):
        return {
            "type": "dict",
            "children": {
                str(key): _flatten_tree(child, arrays, (*path, str(key)))
                for key, child in value.items()
            },
        }
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [
                _flatten_tree(child, arrays, (*path, str(index)))
                for index, child in enumerate(value)
            ],
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "items": [
                _flatten_tree(child, arrays, (*path, str(index)))
                for index, child in enumerate(value)
            ],
        }

    array = np.asarray(value)
    if array.dtype == object:
        raise TypeError(f"object arrays are not supported in checkpoints: {path}")
    key = f"arr_{len(arrays):06d}"
    arrays[key] = array
    return {
        "type": "array",
        "key": key,
        "path": list(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _unflatten_tree(node: Any, archive: Any, expected_keys: set[str]) -> Any:
    if not isinstance(node, dict):
        raise ValueError("checkpoint param_tree nodes must be mappings")
    node_type = node.get("type")
    if node_type == "dict":
        children = node.get("children")
        if not isinstance(children, dict):
            raise ValueError("dict param_tree node must contain children")
        return {
            key: _unflatten_tree(child, archive, expected_keys)
            for key, child in children.items()
        }
    if node_type in {"list", "tuple"}:
        items = node.get("items")
        if not isinstance(items, list):
            raise ValueError(f"{node_type} param_tree node must contain items")
        values = [_unflatten_tree(child, archive, expected_keys) for child in items]
        return tuple(values) if node_type == "tuple" else values
    if node_type == "array":
        key = _required_str(node, "key")
        expected_keys.add(key)
        if key not in archive:
            raise ValueError(f"checkpoint params.npz is missing array: {key}")
        array = archive[key]
        expected_shape = tuple(_required_list(node, "shape"))
        if array.shape != expected_shape:
            raise ValueError(
                f"checkpoint array {key} has shape {array.shape}, "
                f"expected {expected_shape}"
            )
        expected_dtype = _required_str(node, "dtype")
        if str(array.dtype) != expected_dtype:
            raise ValueError(
                f"checkpoint array {key} has dtype {array.dtype}, "
                f"expected {expected_dtype}"
            )
        return np.asarray(array)
    raise ValueError(f"unknown param_tree node type: {node_type!r}")


def _unflatten_optimizer_state(
    raw: dict[str, Any] | None,
    archive: Any,
    expected_keys: set[str],
) -> OptimizerState | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("checkpoint optimizer_state must be a mapping")
    optimizer_type = _required_str(raw, "type")
    slots_tree = _required_dict(raw, "slots_tree")
    return OptimizerState(
        type=optimizer_type,
        step=int(raw["step"]),
        slots=_unflatten_tree(slots_tree, archive, expected_keys),
    )


def _parse_manifest(raw: Any) -> CheckpointManifest:
    if not isinstance(raw, dict):
        raise ValueError("checkpoint manifest must be a mapping")
    manifest = CheckpointManifest(
        schema_version=_required_str(raw, "schema_version"),
        created_by=_required_str(raw, "created_by"),
        student_architecture=_required_str(raw, "student_architecture"),
        student_config=_required_dict(raw, "student_config"),
        step=int(raw["step"]),
        learning_rate=float(raw["learning_rate"]),
        loss_config=_required_dict(raw, "loss_config"),
        target_manifest=_required_dict(raw, "target_manifest"),
        param_tree=_required_dict(raw, "param_tree"),
        optimizer_config=_required_dict(raw, "optimizer_config")
        if "optimizer_config" in raw
        else {},
        optimizer_state=raw.get("optimizer_state"),
        lr_schedule=_required_dict(raw, "lr_schedule") if "lr_schedule" in raw else {},
        notes=_required_string_list(raw.get("notes", []), "notes"),
    )
    validate_checkpoint_manifest(manifest)
    return manifest


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _scalar_int(value: Any) -> int:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError("optimizer_state step must be a scalar")
    return int(array)


def _required_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"checkpoint manifest {key} must be a non-empty string")
    return value


def _required_dict(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint manifest {key} must be a mapping")
    return value


def _required_json_dict(value: Any, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _required_list(raw: dict[str, Any], key: str) -> list[Any]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise ValueError(f"checkpoint manifest {key} must be a list")
    return value


def _required_string_list(value: Any, key: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"checkpoint manifest {key} must be a list of strings")
    return list(value)
