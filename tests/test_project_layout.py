from __future__ import annotations

from pathlib import Path

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
    ROOT / "tests" / "test_imports.py",
    ROOT / "tests" / "test_project_layout.py",
]


def test_required_paths_exist() -> None:
    missing = [path for path in REQUIRED_PATHS if not path.exists()]
    assert not missing, f"Missing required paths: {missing}"
