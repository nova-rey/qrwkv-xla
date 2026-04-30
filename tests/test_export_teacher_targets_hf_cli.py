from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_teacher_targets.py"
CONFIG = ROOT / "configs" / "teacher_export_stub.yaml"


def test_export_teacher_targets_help_lists_hf_flags() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for flag in (
        "--model-id",
        "--tokenizer-id",
        "--prompt",
        "--prompt-file",
        "--trust-remote-code",
        "--revision",
        "--device",
        "--dtype",
    ):
        assert flag in result.stdout


def test_cli_accepts_hf_overrides_without_real_hf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module()
    out_dir = tmp_path / "cli_hf_overrides"

    class StubExporter:
        name = "hf"

        def export(self, request):
            assert request.config.runtime.exporter_backend == "hf"
            assert request.config.teacher.resolved_model_id == "tiny-model"
            assert request.config.teacher.tokenizer_id == "tiny-tokenizer"
            assert request.config.teacher.trust_remote_code is True
            assert request.config.teacher.revision == "test-revision"
            assert request.config.teacher.device == "cpu"
            assert request.config.teacher.dtype == "fp32"
            assert request.config.targets.prompt_texts == ("one", "two")
            manifest = type(
                "Manifest",
                (),
                {"teacher_policy_label": request.config.teacher.policy_label},
            )()
            out_dir.mkdir(parents=True, exist_ok=True)
            return type(
                "Result",
                (),
                {
                    "output_dir": out_dir,
                    "manifest": manifest,
                    "shard_count": 1,
                    "total_examples": 2,
                },
            )()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--config",
            str(CONFIG),
            "--backend",
            "hf",
            "--model-id",
            "tiny-model",
            "--tokenizer-id",
            "tiny-tokenizer",
            "--trust-remote-code",
            "--revision",
            "test-revision",
            "--device",
            "cpu",
            "--dtype",
            "fp32",
            "--prompt",
            "one",
            "--prompt",
            "two",
            "--out",
            str(out_dir),
        ],
    )
    monkeypatch.setattr(
        "qrwkv_xla.teacher_export.get_teacher_exporter",
        lambda _: StubExporter(),
    )
    monkeypatch.setattr(
        "qrwkv_xla.targets.validate_target_bundle",
        lambda _path: None,
    )
    monkeypatch.setattr(
        "qrwkv_xla.targets.inspect_target_bundle",
        lambda _path: {"target_keys": ["attention_mask", "hidden_states", "input_ids"]},
    )

    module.main()

    output = capsys.readouterr().out
    assert "backend: hf" in output
    assert "target_keys: attention_mask, hidden_states, input_ids" in output


def test_cli_hf_backend_missing_dependencies_exits_nonzero(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(CONFIG),
            "--backend",
            "hf",
            "--model-id",
            "tiny-model",
            "--prompt",
            "hello from qrwkv-xla",
            "--out",
            str(tmp_path / "hf_missing"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if 'Install them with: python -m pip install -e ".[teacher-hf]"' in result.stderr:
        assert result.returncode != 0
    else:
        pytest.skip("teacher-hf dependencies are installed in this environment")


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "export_teacher_targets_under_test", SCRIPT
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
