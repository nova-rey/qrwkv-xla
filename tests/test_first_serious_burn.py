from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from qrwkv_xla.burn import (
    FIRST_SERIOUS_BURN_CLAIMS_NOT_MADE,
    FirstSeriousBurnConfig,
    default_first_serious_burn_config,
    run_first_serious_burn,
    write_first_serious_burn_config,
    write_first_serious_burn_report,
)

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_is_conservative_dry_run() -> None:
    config = default_first_serious_burn_config()

    assert config.phase == "P112"
    assert config.mode == "dry_run"
    assert config.local_files_only is True
    assert config.allow_downloads is False
    assert config.max_steps == 1
    assert config.runtime == "reference"


def test_dry_run_executes_without_tpu_gpu_hf_or_internet(tmp_path: Path) -> None:
    readiness_path = _readiness_report(tmp_path, status="pass")
    config = _config(tmp_path, readiness_path=readiness_path)

    result = run_first_serious_burn(config)

    assert result.status == "dry_run_pass"
    assert result.dry_run is True
    assert result.steps_completed == 1
    assert result.readiness_status == "pass"
    assert result.blockers == ()
    assert "qwen_specific_support" in result.claims_not_made


def test_dry_run_writes_report_json(tmp_path: Path) -> None:
    result = run_first_serious_burn(
        _config(tmp_path, readiness_path=_readiness_report(tmp_path, status="pass"))
    )
    report_path = write_first_serious_burn_report(
        result,
        tmp_path / "burn_report.json",
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["phase"] == "P112"
    assert payload["status"] == "dry_run_pass"
    assert payload["mode"] == "dry_run"


def test_dry_run_includes_evidence_paths(tmp_path: Path) -> None:
    result = run_first_serious_burn(
        _config(tmp_path, readiness_path=_readiness_report(tmp_path, status="pass"))
    )

    assert result.preflight_report_path is not None
    assert Path(result.preflight_report_path).is_file()
    assert result.checkpoint_path is not None
    assert Path(result.checkpoint_path).is_dir()
    assert result.eval_report_path is not None
    assert Path(result.eval_report_path).is_file()
    assert result.launch_commands_path is not None
    assert Path(result.launch_commands_path).is_file()


def test_real_mode_without_confirm_is_blocked(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path, readiness_path=_readiness_report(tmp_path, status="pass")),
        mode="real",
    )

    result = run_first_serious_burn(config, confirm_serious_burn=False)

    assert result.status == "blocked"
    assert result.steps_completed == 0
    assert result.blockers == ("real burn mode requires --confirm-serious-burn",)


def test_real_mode_with_failed_readiness_is_blocked(tmp_path: Path) -> None:
    config = replace(
        _config(
            tmp_path,
            readiness_path=_readiness_report(
                tmp_path,
                status="fail",
                blockers=("fixture gate failed",),
            ),
        ),
        mode="real",
    )

    result = run_first_serious_burn(config, confirm_serious_burn=True)

    assert result.status == "blocked"
    assert result.blockers == ("fixture gate failed",)


def test_warn_readiness_requires_accepted_warnings(tmp_path: Path) -> None:
    readiness_path = _readiness_report(
        tmp_path,
        status="warn",
        warnings=("transparent hugepages disabled",),
    )
    blocked = run_first_serious_burn(_config(tmp_path, readiness_path=readiness_path))
    accepted = run_first_serious_burn(
        replace(
            _config(tmp_path / "accepted", readiness_path=readiness_path),
            accepted_warnings=("transparent hugepages disabled",),
        )
    )

    assert blocked.status == "blocked"
    assert blocked.blockers == (
        "P111 readiness report status is warn and warnings were not accepted",
    )
    assert accepted.status == "dry_run_pass"
    assert accepted.warnings == ("transparent hugepages disabled",)


def test_pass_readiness_confirmed_real_mode_requires_textbook(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path, readiness_path=_readiness_report(tmp_path, status="pass")),
        mode="real",
    )

    result = run_first_serious_burn(config, confirm_serious_burn=True)

    assert result.status == "blocked"
    assert result.dry_run is False
    assert result.steps_completed == 0
    assert result.real_training_executed is False
    assert result.blockers == (
        "real mode requires --teacher-textbook or target_store_path",
    )


def test_report_includes_claims_not_made(tmp_path: Path) -> None:
    result = run_first_serious_burn(
        _config(tmp_path, readiness_path=_readiness_report(tmp_path, status="pass"))
    )

    assert result.claims_not_made == FIRST_SERIOUS_BURN_CLAIMS_NOT_MADE
    assert "automatic_burn_launched" in result.claims_not_made
    assert "training_success_guaranteed" in result.claims_not_made


def test_launch_commands_doc_path_is_produced(tmp_path: Path) -> None:
    result = run_first_serious_burn(
        _config(tmp_path, readiness_path=_readiness_report(tmp_path, status="pass"))
    )

    assert result.launch_commands_path is not None
    text = Path(result.launch_commands_path).read_text(encoding="utf-8")
    assert "--confirm-serious-burn" in text
    assert "run_big_burn_readiness_report.py" in text


def test_cli_defaults_to_dry_run_and_writes_report(tmp_path: Path) -> None:
    config_path = tmp_path / "burn_config.json"
    output_dir = tmp_path / "cli_dry_run"
    write_first_serious_burn_config(
        _config(output_dir, readiness_path=_readiness_report(tmp_path, status="pass")),
        config_path,
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_first_serious_burn.py"),
            "--config",
            str(config_path),
            "--output",
            str(output_dir),
        ],
        check=False,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0
    payload = json.loads((output_dir / "burn_report.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "dry_run"
    assert payload["status"] == "dry_run_pass"


def test_cli_exits_nonzero_on_blocked_real_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "burn_config.json"
    output_dir = tmp_path / "cli_real"
    write_first_serious_burn_config(
        replace(
            _config(
                output_dir,
                readiness_path=_readiness_report(tmp_path, status="pass"),
            ),
            mode="real",
        ),
        config_path,
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_first_serious_burn.py"),
            "--config",
            str(config_path),
            "--output",
            str(output_dir),
            "--mode",
            "real",
        ],
        check=False,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    payload = json.loads((output_dir / "burn_report.json").read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert payload["blockers"] == ["real burn mode requires --confirm-serious-burn"]


def test_no_training_at_scale_occurs_in_harness_source() -> None:
    source = (ROOT / "src" / "qrwkv_xla" / "burn" / "first_serious_burn.py").read_text(
        encoding="utf-8"
    )

    assert "run_distill_stage" not in source
    assert "run_lm_stage" not in source
    assert "training_success_guaranteed" in source


def test_no_p112_burn_launches_from_import_or_test(tmp_path: Path) -> None:
    result = run_first_serious_burn(
        _config(tmp_path, readiness_path=_readiness_report(tmp_path, status="pass"))
    )

    assert result.dry_run is True
    assert "automatic_burn_launched" in result.claims_not_made


def _config(tmp_path: Path, *, readiness_path: Path) -> FirstSeriousBurnConfig:
    return replace(
        default_first_serious_burn_config(output_dir=tmp_path),
        readiness_report_path=str(readiness_path),
    )


def _readiness_report(
    tmp_path: Path,
    *,
    status: str,
    blockers: tuple[str, ...] = (),
    warnings: tuple[str, ...] = (),
) -> Path:
    path = tmp_path / f"readiness_{status}_{len(blockers)}_{len(warnings)}.json"
    path.write_text(
        json.dumps(
            {
                "phase": "P111",
                "status": status,
                "blockers": list(blockers),
                "warnings": list(warnings),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
