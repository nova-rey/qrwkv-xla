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
    ROOT / "docs" / "XLA_DISCIPLINE.md",
    ROOT / "docs" / "TPU_SMOKE_GUIDE.md",
    ROOT / "docs" / "DISTILLATION_RUNTIME.md",
    ROOT / "docs" / "HF_TEACHER_EXPORT.md",
    ROOT / "docs" / "NYX_AGENT_ENTRYPOINT.yaml",
    ROOT / "configs" / "tiny_cpu.yaml",
    ROOT / "configs" / "tiny_tpu_smoke.yaml",
    ROOT / "configs" / "tiny_tpu_distill_smoke.yaml",
    ROOT / "configs" / "teacher_export_stub.yaml",
    ROOT / "configs" / "teacher_export_hf_tiny.yaml",
    ROOT / "configs" / "distill_stage0_stub.yaml",
    ROOT / "scripts" / "print_env.py",
    ROOT / "scripts" / "smoke_cpu.py",
    ROOT / "scripts" / "smoke_tpu.py",
    ROOT / "scripts" / "xla_inspect.py",
    ROOT / "scripts" / "tpu_distill_smoke.py",
    ROOT / "scripts" / "run_distill_stage.py",
    ROOT / "src" / "qrwkv_xla" / "distill" / "__init__.py",
    ROOT / "src" / "qrwkv_xla" / "distill" / "config.py",
    ROOT / "src" / "qrwkv_xla" / "distill" / "losses.py",
    ROOT / "src" / "qrwkv_xla" / "distill" / "runner.py",
    ROOT / "src" / "qrwkv_xla" / "distill" / "metrics.py",
    ROOT / "src" / "qrwkv_xla" / "xla" / "__init__.py",
    ROOT / "src" / "qrwkv_xla" / "xla" / "inspect.py",
    ROOT / "src" / "qrwkv_xla" / "xla" / "static_checks.py",
    ROOT / "src" / "qrwkv_xla" / "config" / "schema.py",
    ROOT / "src" / "qrwkv_xla" / "config" / "load.py",
    ROOT / "src" / "qrwkv_xla" / "targets" / "manifest.py",
    ROOT / "src" / "qrwkv_xla" / "targets" / "validate.py",
    ROOT / "src" / "qrwkv_xla" / "teacher_export" / "hf.py",
    ROOT / "src" / "qrwkv_xla" / "teacher_export" / "prompts.py",
    ROOT / "tests" / "test_imports.py",
    ROOT / "tests" / "test_project_layout.py",
    ROOT / "tests" / "test_config_loading.py",
    ROOT / "tests" / "test_target_manifest.py",
    ROOT / "tests" / "test_xla_inspect.py",
    ROOT / "tests" / "test_xla_static_checks.py",
    ROOT / "tests" / "test_tpu_distill_smoke_cli.py",
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
