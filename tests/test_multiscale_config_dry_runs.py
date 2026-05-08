from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from qrwkv_xla.scale_dry_run import (
    P45_HARDWARE_PROFILES,
    P45_TARGET_MODEL_PROFILES,
    checkpoint_skeleton_payload,
    generate_multiscale_configs,
    parameter_band_status,
    parameter_shape_metadata,
    run_shape_dry_run,
    validate_checkpoint_skeleton,
)
from qrwkv_xla.scale_planner import (
    HARDWARE_PROFILES,
    MODEL_PROFILES,
    estimate_qwen_reference_parameters,
)

ROOT = Path(__file__).resolve().parents[1]


def test_p45_profiles_and_hardware_are_present_and_valid() -> None:
    for name in P45_TARGET_MODEL_PROFILES:
        assert name in MODEL_PROFILES
        profile = MODEL_PROFILES[name]
        assert profile.backend == "rwkv7_qwen_reference"
        estimate = estimate_qwen_reference_parameters(profile)
        assert parameter_band_status(name, estimate.total_params)["status"] == "in_band"

    for name in P45_HARDWARE_PROFILES:
        assert name in HARDWARE_PROFILES
        profile = HARDWARE_PROFILES[name]
        assert profile.resolved_per_device_memory_gb > 0
        if profile.device_kind == "tpu":
            assert any(
                "Aggregate" in note or "aggregate" in note for note in profile.notes
            )


def test_p45_preserves_existing_qwen_profile_names() -> None:
    assert MODEL_PROFILES["qwen_0_5b_candidate"].hidden_size == 1024
    assert MODEL_PROFILES["qwen_1_5b_candidate"].hidden_size == 1536
    assert MODEL_PROFILES["qwen_7b_stretch"].hidden_size == 3584


def test_generate_multiscale_configs_writes_reports_and_configs(tmp_path: Path) -> None:
    paths = generate_multiscale_configs(tmp_path)

    for path in paths.values():
        assert path.is_file()
    report = json.loads((tmp_path / "scale_plan_report.json").read_text())
    fit_matrix = json.loads((tmp_path / "fit_matrix.json").read_text())
    markdown = (tmp_path / "P45_SCALE_PLAN_REPORT.md").read_text()

    assert report["status"] == "planning_only_and_dry_run_only"
    assert len(report["plans"]) == len(P45_TARGET_MODEL_PROFILES) * len(
        P45_HARDWARE_PROFILES
    )
    assert set(fit_matrix["models"]) == set(P45_TARGET_MODEL_PROFILES)
    for rows in fit_matrix["models"].values():
        for cell in rows.values():
            assert cell["dry_run_support"] == "metadata_only"
            assert cell["memory_fit"] in {"yes", "maybe", "no", "unknown"}
            assert cell["fit_classification"] in {"yes", "maybe", "no", "unknown"}
            assert cell["recommended_training_mode"] in {
                "hidden_only",
                "hidden_plus_topk_logits",
                "hidden_plus_full_logits",
                "not_recommended",
            }
            assert "component_estimates_bytes" in cell
            assert "estimated_parameter_memory" in cell["component_estimates_bytes"]
            assert "estimated_optimizer_memory" in cell["component_estimates_bytes"]
            assert (
                "activation_sequence_memory_estimate"
                in cell["component_estimates_bytes"]
            )
            assert (
                "target_logits_hidden_target_memory_estimate"
                in cell["component_estimates_bytes"]
            )
            assert "checkpoint_memory_estimate" in cell["component_estimates_bytes"]
            assert "total_estimate" in cell["component_estimates_bytes"]
            assert cell["memory_interpretation"] in {
                "single_device_estimate",
                "aggregate_slice_metadata_only_no_pjit_runtime",
                "aggregate_slice_requires_pjit_sharding",
            }
            assert "sharded_runtime" in cell["warning"] or "estimate" in cell["warning"]
    assert "planning-only and dry-run-only" in markdown
    assert (tmp_path / "P45_RESULTS.md").is_file()

    config = yaml.safe_load(
        (tmp_path / "configs" / "qrwkv_qwen_0_5b_candidate.yaml").read_text()
    )
    assert config["planning_only"] is True
    assert config["distillation"]["student"]["architecture"] == "rwkv7_qwen_reference"


def test_metadata_only_shape_dry_run_does_not_materialize_large_profiles(
    tmp_path: Path,
) -> None:
    payload = run_shape_dry_run(tmp_path)

    assert payload["default_safe"] is True
    for item in payload["dry_runs"]:
        assert item["metadata_only"] is True
        assert item["materialized"] is False
        assert item["checkpoint_skeleton"]["validated"] is True
        assert item["checkpoint_skeleton"]["readback"]["status"] == "valid"
        assert item["fit_by_hardware"]
        assert item["shape_metadata"]["shapes"]["token_embedding.weight"][0] == 151936
        dry_run_path = (
            tmp_path / "dry_runs" / item["model_profile"] / "metadata_dry_run.json"
        )
        assert dry_run_path.is_file()
        skeleton_dir = tmp_path / "checkpoint_skeletons" / item["model_profile"]
        assert (skeleton_dir / "checkpoint_manifest.json").is_file()
        assert (skeleton_dir / "model_config.yaml").is_file()
        assert (skeleton_dir / "checkpoint_metadata.json").is_file()
    assert (tmp_path / "multiscale_shape_dry_run_report.json").is_file()
    assert (tmp_path / "P45_DRY_RUN_REPORT.md").is_file()
    assert (tmp_path / "P45_RESULTS.md").is_file()


def test_tiny_debug_materialization_is_allowed_but_large_is_blocked(
    tmp_path: Path,
) -> None:
    tiny = run_shape_dry_run(
        tmp_path / "tiny",
        model_profiles=("tiny_debug",),
        materialize_init=True,
    )
    assert tiny["dry_runs"][0]["materialization_status"] == "materialized"

    large = run_shape_dry_run(
        tmp_path / "large",
        model_profiles=("qrwkv_qwen_0_5b_candidate",),
        materialize_init=True,
    )
    assert large["dry_runs"][0]["materialized"] is False
    assert large["dry_runs"][0]["materialization_status"].startswith("blocked")


def test_checkpoint_skeleton_is_metadata_first_and_readback_validated(
    tmp_path: Path,
) -> None:
    profile = MODEL_PROFILES["qrwkv_qwen_1_5b_candidate"]
    estimate = estimate_qwen_reference_parameters(profile)
    skeleton = checkpoint_skeleton_payload(profile, estimate.total_params)
    path = tmp_path / "checkpoint_skeleton.json"
    path.write_text(json.dumps(skeleton), encoding="utf-8")

    assert validate_checkpoint_skeleton(path) is True
    bundle = run_shape_dry_run(
        tmp_path / "bundle",
        model_profiles=("qrwkv_qwen_1_5b_candidate",),
    )["dry_runs"][0]["checkpoint_skeleton"]
    assert bundle["readback"]["status"] == "valid"
    assert skeleton["status"] == "metadata_only_checkpoint_skeleton"
    assert "token_embedding.weight" in skeleton["param_shapes"]


def test_shape_metadata_tracks_qwen_reference_surface() -> None:
    profile = MODEL_PROFILES["qrwkv_qwen_0_5b_candidate"]
    shapes = parameter_shape_metadata(profile)["shapes"]

    assert shapes["layers.self_attn.q_proj.weight"] == [24, 1024, 1024]
    assert shapes["layers.self_attn.k_proj.weight"] == [24, 1024, 512]
    assert shapes["layers.mlp.down_proj.weight"] == [24, 2816, 1024]


def test_p45_cli_help_and_import_behavior() -> None:
    for script in (
        "generate_multiscale_configs.py",
        "run_multiscale_shape_dry_run.py",
    ):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout


def test_p45_cli_commands_write_default_safe_outputs(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_multiscale_configs.py"),
            "--out",
            str(config_dir),
            "--overwrite",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (config_dir / "scale_plan_report.json").is_file()
    assert "planning-only" in result.stdout

    dry_dir = tmp_path / "dry"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_multiscale_shape_dry_run.py"),
            "--scale-plan",
            str(config_dir / "scale_plan_report.json"),
            "--metadata-only",
            "--out",
            str(dry_dir),
            "--profiles",
            "qrwkv_qwen_0_5b_candidate",
            "--overwrite",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads((dry_dir / "multiscale_shape_dry_run_report.json").read_text())
    assert payload["default_safe"] is True
    assert all(not item["materialized"] for item in payload["dry_runs"])
    assert (dry_dir / "dry_runs" / "qrwkv_qwen_0_5b_candidate").is_dir()
