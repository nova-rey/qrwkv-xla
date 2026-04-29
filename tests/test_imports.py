from __future__ import annotations

import importlib

MODULES = [
    "qrwkv_xla",
    "qrwkv_xla.config",
    "qrwkv_xla.config.load",
    "qrwkv_xla.config.schema",
    "qrwkv_xla.teacher_export",
    "qrwkv_xla.targets",
    "qrwkv_xla.targets.manifest",
    "qrwkv_xla.targets.validate",
    "qrwkv_xla.students",
    "qrwkv_xla.trainers",
    "qrwkv_xla.losses",
    "qrwkv_xla.eval",
    "qrwkv_xla.checkpointing",
    "qrwkv_xla.utils",
]


def test_imports() -> None:
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        assert module is not None
