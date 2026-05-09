from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax
import numpy as np

from qrwkv_xla.parity.radlads_parameter_import import (
    import_radlads_parameters_for_replay,
)
from qrwkv_xla.parity.radlads_parameter_mapping import (
    normalize_radlads_parameter_arrays,
)
from qrwkv_xla.parity.radlads_replay import replay_radlads_tiny_numerical_fixtures
from qrwkv_xla.students import RWKV7QwenReferenceConfig, RWKV7QwenReferenceStudent

ROOT = Path(__file__).resolve().parents[1]
P49_FIXTURES = ROOT / "artifacts" / "p49_radlads_numerical_parity" / "radlads_fixtures"


def test_qwen_reference_qkv_bias_and_replay_gate_are_explicit() -> None:
    config = RWKV7QwenReferenceConfig(
        vocab_size=16,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        intermediate_size=16,
        emit_logits=True,
        use_rope=False,
        radlads_replay_mode=True,
        radlads_compatible_math=True,
        radlads_attention_group_norm=True,
        radlads_balance_state=True,
        lora_rank_decay=4,
        lora_rank_iclr=4,
        lora_rank_value_residual_mix=4,
        lora_rank_gate=4,
    )
    params = RWKV7QwenReferenceStudent(config).init_params(jax.random.PRNGKey(1))
    self_attn = params["layers"]["self_attn"]

    assert self_attn["q_proj"]["bias"].shape == (1, 8)
    assert self_attn["k_proj"]["bias"].shape == (1, 4)
    assert self_attn["v_proj"]["bias"].shape == (1, 4)
    assert self_attn["g1"].shape == (1, 8, 4)
    assert self_attn["g2"].shape == (1, 4, 8)


def test_qkv_bias_changes_output_deterministically() -> None:
    base = RWKV7QwenReferenceConfig(
        vocab_size=16,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        intermediate_size=16,
        emit_logits=True,
        use_rope=False,
        attention_qkv_bias=False,
    )
    biased = RWKV7QwenReferenceConfig(
        **{**base.__dict__, "attention_qkv_bias": True},
    )
    base_student = RWKV7QwenReferenceStudent(base)
    biased_student = RWKV7QwenReferenceStudent(biased)
    input_ids = np.array([[1, 2, 3]], dtype=np.int32)

    base_params = base_student.init_params(jax.random.PRNGKey(7))
    biased_params = biased_student.init_params(jax.random.PRNGKey(7))
    biased_params["layers"]["self_attn"]["q_proj"]["bias"] = jnp_full((1, 8), 0.25)
    biased_params["layers"]["self_attn"]["k_proj"]["bias"] = jnp_full((1, 4), -0.5)
    biased_params["layers"]["self_attn"]["v_proj"]["bias"] = jnp_full((1, 4), 0.75)

    base_out, _ = base_student.apply_with_state(base_params, input_ids)
    biased_out, _ = biased_student.apply_with_state(biased_params, input_ids)

    assert not np.allclose(
        np.asarray(base_out.hidden_states),
        np.asarray(biased_out.hidden_states),
    )


def test_radlads_parameter_import_reports_defaulted_and_excluded_surfaces() -> None:
    if not (P49_FIXTURES / "radlads_parameters.npz").exists():
        return

    result = import_radlads_parameters_for_replay(
        P49_FIXTURES / "radlads_parameters.npz",
        manifest_path=P49_FIXTURES / "manifest.json",
        allow_defaults=True,
    )

    assert result.qrwkv_config.radlads_replay_mode is True
    assert result.overall_status == "pass"
    assert len(result.mapped) >= 20
    assert len(result.defaulted) >= 6
    assert len(result.excluded) >= 1
    defaulted = {row["qrwkv"] for row in result.defaulted}
    assert "layers.self_attn.g_proj.weight" in defaulted
    assert "lm_head.bias" in defaulted
    assert result.report["g1_g2_status"]["status"] == "implemented"
    assert result.report["qkv_bias_status"]["status"] == "implemented"


def test_normalized_radlads_parameters_include_qkv_bias_and_g1_g2() -> None:
    if not (P49_FIXTURES / "radlads_parameters.npz").exists():
        return

    with np.load(P49_FIXTURES / "radlads_parameters.npz") as payload:
        normalized = normalize_radlads_parameter_arrays(
            {name: payload[name] for name in payload.files}
        )

    assert normalized["layers.self_attn.q_proj.bias"].shape == (2, 8)
    assert normalized["layers.self_attn.k_proj.bias"].shape == (2, 4)
    assert normalized["layers.self_attn.v_proj.bias"].shape == (2, 4)
    assert normalized["layers.self_attn.g1"].shape == (2, 8, 4)
    assert normalized["layers.self_attn.g2"].shape == (2, 4, 8)


def test_replay_attempts_real_p49_comparisons_when_fixture_exists(
    tmp_path: Path,
) -> None:
    manifest = P49_FIXTURES / "manifest.json"
    if not manifest.exists():
        return

    report = replay_radlads_tiny_numerical_fixtures(
        manifest,
        out_dir=tmp_path / "replay",
    )

    assert report["import_report"]["overall_status"] == "pass"
    assert report["attempted_comparisons"] > 0
    statuses = {
        row["status"] for case in report["cases"] for row in case.get("comparisons", [])
    }
    assert "unsupported" not in statuses
    assert statuses & {"pass", "fail", "non_finite", "shape_mismatch", "dtype_mismatch"}
    by_name = {case["name"]: case for case in report["cases"]}
    tiny = by_name["tiny_no_mask"]
    assert tiny["replay_profile"]["all_radlads_math"] is False
    assert all(row["status"] != "non_finite" for row in tiny["comparisons"])
    assert (tmp_path / "replay" / "replay_comparison_report.json").is_file()
    assert (tmp_path / "replay" / "parameter_import_report.json").is_file()
    assert (tmp_path / "replay" / "P50_RESULTS.md").is_file()
    assert (tmp_path / "replay" / "P50_SURFACE_COMPARISON.md").is_file()


def test_replay_cli_help() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "replay_radlads_tiny_numerical_fixtures.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "P50" in result.stdout
    assert "--overwrite" in result.stdout


def jnp_full(shape: tuple[int, ...], value: float):
    import jax.numpy as jnp

    return jnp.full(shape, value, dtype=jnp.float32)
