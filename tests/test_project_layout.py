from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    ROOT / "README.md",
    ROOT / "pyproject.toml",
    ROOT / "docs" / "QRWKV_SNAPSHOT.yaml",
    ROOT / "docs" / "QRWKV_BIBLE.md",
    ROOT / "docs" / "ARCHITECTURE.md",
    ROOT / "docs" / "ROADMAP.md",
    ROOT / "docs" / "DECISIONS.md",
    ROOT / "docs" / "RISK_REGISTER.md",
    ROOT / "docs" / "ARTIFACT_FORMATS.md",
    ROOT / "docs" / "TESTING_STRATEGY.md",
    ROOT / "docs" / "TPU_NOTES.md",
    ROOT / "docs" / "NYX_AGENT_ENTRYPOINT.yaml",
    ROOT / "configs" / "tiny_cpu.yaml",
    ROOT / "configs" / "tiny_tpu_smoke.yaml",
    ROOT / "configs" / "teacher_export_stub.yaml",
    ROOT / "configs" / "distill_stage0_stub.yaml",
    ROOT / "scripts" / "print_env.py",
    ROOT / "scripts" / "smoke_cpu.py",
    ROOT / "scripts" / "smoke_tpu.py",
    ROOT / "scripts" / "run_distill_stage.py",
    ROOT / "docs" / "DISTILLATION_RUNTIME.md",
    ROOT / "src" / "qrwkv_xla" / "distill" / "__init__.py",
    ROOT / "src" / "qrwkv_xla" / "distill" / "config.py",
    ROOT / "src" / "qrwkv_xla" / "distill" / "losses.py",
    ROOT / "src" / "qrwkv_xla" / "distill" / "runner.py",
    ROOT / "src" / "qrwkv_xla" / "distill" / "metrics.py",
    ROOT / "src" / "qrwkv_xla" / "config" / "schema.py",
    ROOT / "src" / "qrwkv_xla" / "config" / "load.py",
    ROOT / "src" / "qrwkv_xla" / "targets" / "manifest.py",
    ROOT / "src" / "qrwkv_xla" / "targets" / "validate.py",
    ROOT / "tests" / "test_imports.py",
    ROOT / "tests" / "test_project_layout.py",
    ROOT / "tests" / "test_config_loading.py",
    ROOT / "tests" / "test_target_manifest.py",
]


def test_required_paths_exist() -> None:
    missing = [path for path in REQUIRED_PATHS if not path.exists()]
    assert not missing, f"Missing required paths: {missing}"


def test_yaml_docs_parse() -> None:
    for path in [
        ROOT / "docs" / "QRWKV_SNAPSHOT.yaml",
        ROOT / "docs" / "NYX_AGENT_ENTRYPOINT.yaml",
    ]:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
