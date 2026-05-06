from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.generate_qwen_reference_fixtures import write_fixtures

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "qwen_reference"
ATOL = 1e-5
RTOL = 1e-5


def test_checked_in_qwen_reference_fixture_manifest_is_complete() -> None:
    manifest = _read_manifest(FIXTURE_DIR)

    assert manifest["backend"] == "rwkv7_qwen_reference"
    assert manifest["seed"] == 1234
    assert manifest["dtype"] == "float32"
    assert manifest["config"] == {
        "batch_size": 2,
        "emit_logits": True,
        "emit_mixer_outputs": True,
        "hidden_size": 8,
        "num_heads": 2,
        "num_kv_heads": 1,
        "num_layers": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "sequence_length": 5,
        "tie_embeddings": False,
        "vocab_size": 32,
    }
    assert {case["name"] for case in manifest["cases"]} == {
        "no_mask",
        "interior_mask",
        "prefix_left_padding",
    }
    behavior = manifest["masked_token_behavior"]
    assert "shift_state is updated" in behavior["summary"]
    assert "positions still consume positions" in behavior["left_padding"]


def test_qwen_reference_fixture_payloads_lock_full_vs_stepwise_parity() -> None:
    manifest = _read_manifest(FIXTURE_DIR)

    for case in manifest["cases"]:
        payload = _load_case(case)
        assert case["payload_sha256"] == _hash_npz_payload(payload)

        assert payload["input_ids"].shape == (2, 5)
        assert payload["full_hidden_states"].shape == (2, 2, 5, 8)
        assert payload["full_logits"].shape == (2, 5, 32)
        assert payload["full_mixer_outputs"].shape == (2, 2, 5, 8)
        assert payload["full_wkv_matrix_state"].shape == (2, 2, 2, 4, 4)
        assert payload["full_shift_state"].shape == (2, 2, 8)
        assert int(payload["full_next_position"]) == 5
        assert int(payload["step_next_position"]) == 5

        np.testing.assert_allclose(
            payload["full_hidden_states"],
            payload["step_hidden_states"],
            atol=ATOL,
            rtol=RTOL,
        )
        np.testing.assert_allclose(
            payload["full_logits"], payload["step_logits"], atol=ATOL, rtol=RTOL
        )
        np.testing.assert_allclose(
            payload["full_mixer_outputs"],
            payload["step_mixer_outputs"],
            atol=ATOL,
            rtol=RTOL,
        )
        np.testing.assert_allclose(
            payload["full_wkv_matrix_state"],
            payload["step_wkv_matrix_state"],
            atol=ATOL,
            rtol=RTOL,
        )
        np.testing.assert_allclose(
            payload["full_shift_state"],
            payload["step_shift_state"],
            atol=ATOL,
            rtol=RTOL,
        )
        assert all(value <= ATOL for value in case["equivalence"].values())


def test_qwen_reference_mask_edge_fixture_behavior_is_documented() -> None:
    manifest = _read_manifest(FIXTURE_DIR)
    by_name = {case["name"]: case for case in manifest["cases"]}

    no_mask = _load_case(by_name["no_mask"])
    assert "attention_mask" not in no_mask

    interior = _load_case(by_name["interior_mask"])
    assert interior["attention_mask"].tolist() == [
        [1, 1, 0, 1, 1],
        [1, 0, 1, 1, 1],
    ]
    for batch, token in ((0, 2), (1, 1)):
        np.testing.assert_allclose(
            interior["full_mixer_outputs"][batch, :, token, :],
            0.0,
            atol=ATOL,
            rtol=RTOL,
        )
        assert np.max(np.abs(interior["full_logits"][batch, token, :])) > 0.0

    prefix = _load_case(by_name["prefix_left_padding"])
    assert prefix["attention_mask"].tolist() == [
        [0, 0, 1, 1, 1],
        [0, 1, 1, 1, 1],
    ]
    assert by_name["prefix_left_padding"]["attention_mask"]["kind"] == (
        "prefix_left_padding"
    )
    assert int(prefix["full_next_position"]) == 5


def test_qwen_reference_parameter_surface_snapshot_is_stable() -> None:
    manifest = _read_manifest(FIXTURE_DIR)
    surface = manifest["parameter_surface"]
    paths = [leaf["path"] for leaf in surface["leaves"]]

    assert surface["leaf_count"] == 19
    assert surface["sha256"] == _hash_param_surface(surface["leaves"])
    assert paths == [
        "final_layernorm.weight",
        "layers.input_layernorm.weight",
        "layers.mlp.down_proj.weight",
        "layers.mlp.gate_proj.weight",
        "layers.mlp.up_proj.weight",
        "layers.post_attention_layernorm.weight",
        "layers.self_attn.a_proj.weight",
        "layers.self_attn.b_proj.weight",
        "layers.self_attn.g_proj.weight",
        "layers.self_attn.k_proj.weight",
        "layers.self_attn.o_proj.weight",
        "layers.self_attn.q_proj.weight",
        "layers.self_attn.time_bias",
        "layers.self_attn.time_mix",
        "layers.self_attn.v_proj.weight",
        "layers.self_attn.w_proj.weight",
        "lm_head.bias",
        "lm_head.weight",
        "token_embedding.weight",
    ]
    shapes = {leaf["path"]: leaf["shape"] for leaf in surface["leaves"]}
    assert shapes["token_embedding.weight"] == [32, 8]
    assert shapes["layers.self_attn.k_proj.weight"] == [2, 8, 4]
    assert shapes["layers.self_attn.q_proj.weight"] == [2, 8, 8]
    assert shapes["layers.mlp.down_proj.weight"] == [2, 32, 8]
    assert shapes["lm_head.bias"] == [32]
    assert shapes["lm_head.weight"] == [8, 32]
    assert not any("toy" in path or "shell" in path for path in paths)


def test_qwen_reference_fixture_generation_is_deterministic(tmp_path: Path) -> None:
    generated = tmp_path / "qwen_reference"
    write_fixtures(generated, seed=1234, overwrite=True)

    checked_in = _read_manifest(FIXTURE_DIR)
    fresh = _read_manifest(generated)

    assert _manifest_without_runtime(fresh) == _manifest_without_runtime(checked_in)
    for case in fresh["cases"]:
        generated_payload = _load_case(case, base=generated)
        assert case["payload_sha256"] == _hash_npz_payload(generated_payload)


def test_qwen_reference_fixture_script_smoke(tmp_path: Path) -> None:
    out = tmp_path / "scripted"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_qwen_reference_fixtures.py"),
            "--out",
            str(out),
            "--seed",
            "1234",
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = _read_manifest(out)
    assert "wrote 3 qwen reference fixture cases" in result.stdout
    assert (out / "manifest.json").is_file()
    assert _manifest_without_runtime(manifest) == _manifest_without_runtime(
        _read_manifest(FIXTURE_DIR)
    )
    for case in manifest["cases"]:
        assert case["payload_sha256"] == _hash_npz_payload(_load_case(case, base=out))


def _read_manifest(base: Path) -> dict:
    return json.loads((base / "manifest.json").read_text(encoding="utf-8"))


def _load_case(case: dict, *, base: Path = FIXTURE_DIR) -> dict[str, np.ndarray]:
    with np.load(base / case["payload"]) as payload:
        return {name: payload[name] for name in payload.files}


def _manifest_without_runtime(manifest: dict) -> dict:
    stable = dict(manifest)
    stable["jax"] = {"default_backend": manifest["jax"]["default_backend"]}
    stable["cases"] = [
        {key: value for key, value in case.items() if key != "payload_sha256"}
        for case in manifest["cases"]
    ]
    return stable


def _hash_npz_payload(payload: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(payload):
        array = np.asarray(payload[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(json.dumps(list(array.shape)).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _hash_param_surface(surface: list[dict]) -> str:
    payload = json.dumps(surface, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
