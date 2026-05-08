from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from qrwkv_xla.parity import build_parameter_surface_map
from qrwkv_xla.students import RWKV7QwenReferenceConfig, RWKV7QwenReferenceStudent

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

ROOT = Path(__file__).resolve().parents[1]


def test_radlads_lora_config_validation_and_effective_flags() -> None:
    legacy = RWKV7QwenReferenceConfig(hidden_size=8, num_heads=2)
    assert not legacy.use_radlads_low_rank_decay
    assert not legacy.use_radlads_low_rank_iclr
    assert not legacy.use_radlads_value_residual_mix
    assert not legacy.use_radlads_balance_state_terms
    assert not legacy.use_radlads_attention_group_norm

    compat = RWKV7QwenReferenceConfig(
        hidden_size=8,
        num_heads=2,
        radlads_compatible_math=True,
    )
    assert compat.use_radlads_low_rank_decay
    assert compat.use_radlads_low_rank_iclr
    assert compat.use_radlads_value_residual_mix
    assert compat.use_radlads_balance_state_terms
    assert compat.use_radlads_attention_output_scale
    assert not compat.use_radlads_attention_group_norm
    assert compat.effective_lora_rank_decay == 32

    with pytest.raises(ValueError, match="lora_rank_decay"):
        RWKV7QwenReferenceConfig(
            hidden_size=8,
            num_heads=2,
            lora_rank_decay=0,
        )
    with pytest.raises(ValueError, match="radlads_balance_state requires"):
        RWKV7QwenReferenceConfig(
            hidden_size=8,
            num_heads=2,
            radlads_balance_state=True,
        )


def test_radlads_lora_parameter_leaves_are_present() -> None:
    student = _student(
        radlads_compatible_math=True,
        radlads_attention_group_norm=True,
    )
    params = student.init_params(jax.random.PRNGKey(0))
    attn = params["layers"]["self_attn"]

    assert attn["w0"].shape == (2, 8)
    assert attn["w1"].shape == (2, 8, 4)
    assert attn["w2"].shape == (2, 4, 8)
    assert attn["a0"].shape == (2, 8)
    assert attn["a1"].shape == (2, 8, 4)
    assert attn["a2"].shape == (2, 4, 8)
    assert attn["v0"].shape == (2, 8)
    assert attn["v1"].shape == (2, 8, 4)
    assert attn["v2"].shape == (2, 4, 8)
    assert attn["k_k"].shape == (2, 8)
    assert attn["k_a"].shape == (2, 8)
    assert attn["r_k"].shape == (2, 2, 4)
    assert attn["ln_x"]["weight"].shape == (2, 8)
    assert attn["ln_x"]["bias"].shape == (2, 8)
    assert _tree_is_finite(params)


def test_legacy_default_forward_is_preserved_against_disabled_flags() -> None:
    key = jax.random.PRNGKey(1)
    legacy = _student(radlads_compatible_math=False)
    explicit_legacy = _student(
        radlads_compatible_math=False,
        radlads_low_rank_decay=False,
        radlads_low_rank_iclr=False,
        radlads_value_residual_mix=False,
        radlads_balance_state_terms=False,
        radlads_attention_group_norm=False,
    )
    params = legacy.init_params(key)
    input_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)

    first, first_state = legacy.apply_with_state(params, input_ids)
    second, second_state = explicit_legacy.apply_with_state(params, input_ids)

    assert jnp.max(jnp.abs(first.hidden_states - second.hidden_states)) < 1e-6
    state_delta = first_state.wkv_matrix_state - second_state.wkv_matrix_state
    assert jnp.max(jnp.abs(state_delta)) < 1e-6


def test_radlads_compatible_forward_is_finite_and_stepwise_equivalent() -> None:
    student = _student(radlads_compatible_math=True)
    params = student.init_params(jax.random.PRNGKey(2))
    input_ids = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    attention_mask = jnp.array([[1, 1, 1, 1]], dtype=jnp.int32)

    full_output, full_state = student.apply_with_state(
        params,
        input_ids,
        attention_mask=attention_mask,
    )

    state = student.init_state(batch_size=1)
    hidden_steps = []
    for index in range(input_ids.shape[1]):
        token_output, state = student.step(
            params,
            input_ids[:, index : index + 1],
            state,
            attention_mask=attention_mask[:, index : index + 1],
        )
        hidden_steps.append(token_output.hidden_states)

    step_hidden = jnp.concatenate(hidden_steps, axis=2)
    assert _tree_is_finite(full_output)
    assert _tree_is_finite(full_state)
    assert jnp.max(jnp.abs(full_output.hidden_states - step_hidden)) < 1e-5
    assert jnp.max(jnp.abs(full_state.wkv_matrix_state - state.wkv_matrix_state)) < 1e-5


def test_radlads_attention_group_norm_flag_changes_enabled_path() -> None:
    base = _student(radlads_low_rank_decay=True)
    normed = _student(
        radlads_low_rank_decay=True,
        radlads_attention_group_norm=True,
    )
    base_params = base.init_params(jax.random.PRNGKey(3))
    normed_params = normed.init_params(jax.random.PRNGKey(3))
    input_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)

    base_output = base.apply(base_params, input_ids)
    normed_output = normed.apply(normed_params, input_ids)

    assert base_output.hidden_states.shape == normed_output.hidden_states.shape
    assert _tree_is_finite(normed_output)
    assert not jnp.allclose(base_output.hidden_states, normed_output.hidden_states)


def test_parameter_map_updates_for_p48_surfaces() -> None:
    rows = {row["radlads"]: row for row in build_parameter_surface_map()["mappings"]}

    assert rows["layers.*.self_attn.w0/w1/w2"]["status"] == "represented_flagged_math"
    assert rows["layers.*.self_attn.a0/a1/a2"]["status"] == "represented_flagged_math"
    assert rows["layers.*.self_attn.v0/v1/v2"]["status"] == "represented_flagged_math"
    assert rows["layers.*.self_attn.ln_x"]["status"] == "represented_flagged_math"
    assert rows["layers.*.self_attn.k_k/k_a/r_k"]["status"] == (
        "partially_represented_flagged_math"
    )
    assert "commented out" in rows["layers.*.self_attn.k_k/k_a/r_k"]["notes"]


def test_p48_smoke_script_writes_expected_reports(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_radlads_lora_rank_math_smoke.py"),
            "--out-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "status=pass" in result.stdout
    report = json.loads((tmp_path / "lora_rank_math_report.json").read_text())
    assert report["status"] == "pass"
    assert (tmp_path / "P48_RESULTS.md").is_file()
    assert (tmp_path / "P48_PARAMETER_SURFACE_MAP.md").is_file()
    assert (tmp_path / "parameter_surface_map.json").is_file()


def _student(**kwargs: object) -> RWKV7QwenReferenceStudent:
    return RWKV7QwenReferenceStudent(
        RWKV7QwenReferenceConfig(
            vocab_size=17,
            hidden_size=8,
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            emit_logits=True,
            lora_rank_decay=4,
            lora_rank_iclr=4,
            lora_rank_value_residual_mix=4,
            **kwargs,
        )
    )


def _tree_is_finite(tree: object) -> bool:
    return all(
        bool(jnp.all(jnp.isfinite(jnp.asarray(leaf))))
        for leaf in jax.tree_util.tree_leaves(tree)
    )
