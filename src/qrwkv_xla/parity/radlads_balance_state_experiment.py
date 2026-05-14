from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from qrwkv_xla.parity.radlads_replay_diagnostics import summarize_array
from qrwkv_xla.parity.radlads_wkv_trace import WKVTraceCollector
from qrwkv_xla.students import RWKV7QwenReferenceConfig, RWKV7QwenReferenceStudent

BALANCE_STATE_EXPERIMENT_SCHEMA = "qrwkv_xla.p65_balance_state_experiment.v1"
BALANCE_STATE_STABILITY_SCHEMA = "qrwkv_xla.p65_balance_state_stability.v1"
DEFAULT_EXPERIMENT_OUT = Path("artifacts/p65_balance_state_experiment")
DEFAULT_FIXTURE_MANIFEST = Path("tests/fixtures/radlads_source_parity/manifest.json")

SURFACE_NAMES = (
    "log_w",
    "logits",
    "hidden_states",
    "wkv_matrix_state",
    "shift_state",
)


def base_balance_state_experiment_config() -> RWKV7QwenReferenceConfig:
    return RWKV7QwenReferenceConfig(
        vocab_size=32,
        hidden_size=8,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        emit_logits=True,
        emit_mixer_outputs=True,
        lora_rank_decay=4,
        lora_rank_iclr=4,
        lora_rank_value_residual_mix=4,
    )


def experimental_balance_state_config(
    config: RWKV7QwenReferenceConfig,
) -> RWKV7QwenReferenceConfig:
    return replace(
        config,
        radlads_balance_state_terms=True,
        radlads_balance_state=True,
    )


def run_balance_state_experiment(
    *,
    out_dir: Path = DEFAULT_EXPERIMENT_OUT,
    fixture_manifest: Path = DEFAULT_FIXTURE_MANIFEST,
    seed: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    _prepare_out_dir(out_dir, overwrite=overwrite)
    manifest = _load_manifest(fixture_manifest)
    run_seed = int(seed if seed is not None else manifest.get("seed", 4040))
    base_config = base_balance_state_experiment_config()
    off = _run_mode(
        mode="off",
        config=base_config,
        fixture_manifest=fixture_manifest,
        seed=run_seed,
        out_dir=out_dir / "off",
    )
    experimental = _run_mode(
        mode="experimental",
        config=experimental_balance_state_config(base_config),
        fixture_manifest=fixture_manifest,
        seed=run_seed,
        out_dir=out_dir / "experimental",
    )
    report = _compare_modes(
        off=off,
        experimental=experimental,
        seed=run_seed,
        fixture_manifest=fixture_manifest,
    )
    (out_dir / "balance_state_experiment_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "P65_RESULTS.md").write_text(_experiment_markdown(report), "utf-8")
    (out_dir / "DIFF_SUMMARY.md").write_text(_diff_markdown(report), "utf-8")
    (out_dir / "OFF_VS_EXPERIMENTAL.md").write_text(
        _off_vs_experimental_markdown(report),
        "utf-8",
    )
    return report


def run_balance_state_stability_smoke(
    *,
    out_dir: Path = DEFAULT_EXPERIMENT_OUT,
    seed: int = 6500,
    overwrite: bool = False,
) -> dict[str, Any]:
    stability_dir = out_dir / "stability"
    _prepare_out_dir(stability_dir, overwrite=overwrite)
    input_ids = jnp.array([[1, 2, 3], [3, 2, 1]], dtype=jnp.int32)
    attention_mask = jnp.array([[1, 1, 1], [1, 0, 1]], dtype=jnp.int32)
    config = experimental_balance_state_config(base_balance_state_experiment_config())
    student = RWKV7QwenReferenceStudent(config)
    params = student.init_params(jax.random.PRNGKey(seed))
    full_output, full_state = student.apply_with_state(
        params,
        input_ids,
        attention_mask=attention_mask,
    )
    state = student.init_state(batch_size=int(input_ids.shape[0]))
    hidden_steps = []
    logits_steps = []
    for index in range(int(input_ids.shape[1])):
        token_output, state = student.step(
            params,
            input_ids[:, index : index + 1],
            state,
            attention_mask=attention_mask[:, index : index + 1],
        )
        hidden_steps.append(token_output.hidden_states)
        if token_output.logits is not None:
            logits_steps.append(token_output.logits)
    step_hidden = jnp.concatenate(hidden_steps, axis=2)
    step_logits = jnp.concatenate(logits_steps, axis=1) if logits_steps else None
    hidden_delta = _max_abs(full_output.hidden_states, step_hidden)
    logits_delta = (
        _max_abs(full_output.logits, step_logits)
        if full_output.logits is not None and step_logits is not None
        else 0.0
    )
    state_delta = _max_abs(full_state.wkv_matrix_state, state.wkv_matrix_state)
    finite = all(
        _finite_status(value)["nonfinite_count"] == 0
        for value in (
            full_output.hidden_states,
            full_output.logits,
            full_state.wkv_matrix_state,
            full_state.shift_state,
            state.wkv_matrix_state,
            state.shift_state,
        )
        if value is not None
    )
    report: dict[str, Any] = {
        "schema": BALANCE_STATE_STABILITY_SCHEMA,
        "phase": "P65",
        "status": "pass"
        if finite and max(hidden_delta, logits_delta, state_delta) < 1e-5
        else "fail",
        "mode": "experimental",
        "experimental_flags": _mode_flags(config),
        "seed": seed,
        "finite": finite,
        "full_vs_stepwise": {
            "hidden_states_max_abs": hidden_delta,
            "logits_max_abs": logits_delta,
            "wkv_matrix_state_max_abs": state_delta,
        },
        "surfaces": {
            "hidden_states": _summary("hidden_states", full_output.hidden_states),
            "logits": _summary("logits", full_output.logits),
            "wkv_matrix_state": _summary(
                "wkv_matrix_state", full_state.wkv_matrix_state
            ),
            "shift_state": _summary("shift_state", full_state.shift_state),
        },
        "notes": [
            "CPU/local tiny smoke only.",
            "The experimental balance-state path remains opt-in.",
        ],
    }
    (stability_dir / "stability_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "STABILITY_SMOKE.md").write_text(_stability_markdown(report), "utf-8")
    return report


def _run_mode(
    *,
    mode: str,
    config: RWKV7QwenReferenceConfig,
    fixture_manifest: Path,
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(fixture_manifest)
    student = RWKV7QwenReferenceStudent(config)
    params = student.init_params(jax.random.PRNGKey(seed))
    cases = []
    for case in manifest["cases"]:
        arrays = _load_case_arrays(fixture_manifest, case)
        collector = WKVTraceCollector(case=case["name"], side=mode, include_arrays=True)
        input_ids = jnp.asarray(arrays["input_ids"], dtype=jnp.int32)
        attention_mask = (
            jnp.asarray(arrays["attention_mask"], dtype=jnp.int32)
            if "attention_mask" in arrays
            else None
        )
        output, state = student.apply_with_state(
            params,
            input_ids,
            attention_mask=attention_mask,
            diagnostics=collector,
        )
        surfaces = {
            "log_w": _stack_trace_stage(collector.entries, "log_w"),
            "logits": output.logits,
            "hidden_states": output.hidden_states,
            "wkv_matrix_state": state.wkv_matrix_state,
            "shift_state": state.shift_state,
        }
        cases.append(
            {
                "name": case["name"],
                "attention_mask": case.get("attention_mask", {}),
                "surfaces": {
                    name: _summary(name, value) for name, value in surfaces.items()
                },
                "arrays": {
                    name: _json_array(value) for name, value in surfaces.items()
                },
                "trace_entries": collector.entries,
            }
        )
    payload = {
        "schema": BALANCE_STATE_EXPERIMENT_SCHEMA + ".mode",
        "phase": "P65",
        "mode": mode,
        "mode_status": "experimental" if mode == "experimental" else "off",
        "experimental_flags": _mode_flags(config),
        "seed": seed,
        "cases": cases,
    }
    (out_dir / "mode_report.json").write_text(
        json.dumps(_without_arrays(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "mode_arrays.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _compare_modes(
    *,
    off: dict[str, Any],
    experimental: dict[str, Any],
    seed: int,
    fixture_manifest: Path,
) -> dict[str, Any]:
    cases = []
    for off_case, exp_case in zip(off["cases"], experimental["cases"], strict=True):
        comparisons = []
        for surface in SURFACE_NAMES:
            left = np.asarray(off_case["arrays"][surface], dtype=np.float32)
            right = np.asarray(exp_case["arrays"][surface], dtype=np.float32)
            comparisons.append(_compare_surface(surface, left, right))
        cases.append(
            {
                "name": off_case["name"],
                "status": (
                    "experimental_differs"
                    if any(item["max_abs"] > 0.0 for item in comparisons)
                    else "same"
                ),
                "first_divergent_stage": _first_divergent_stage(
                    off_case["trace_entries"],
                    exp_case["trace_entries"],
                ),
                "comparisons": comparisons,
            }
        )
    return {
        "schema": BALANCE_STATE_EXPERIMENT_SCHEMA,
        "phase": "P65",
        "status": "pass",
        "claim": (
            "Compares default/off QRWKV behavior with the opt-in experimental "
            "RADLADS balance-state compatibility path on tiny local fixtures."
        ),
        "fixture_manifest": str(fixture_manifest),
        "seed": seed,
        "default_behavior_preserved": _off_flags_are_default(off["experimental_flags"]),
        "off_mode_status": off["mode_status"],
        "experimental_mode_status": experimental["mode_status"],
        "off_flags": off["experimental_flags"],
        "experimental_flags": experimental["experimental_flags"],
        "cases": cases,
        "notes": [
            "No tolerance loosening is performed.",
            "P58 log_w is captured as a comparison surface and is not rewritten.",
            "The experimental path is not promoted to default.",
        ],
    }


def _compare_surface(name: str, left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    diff = np.abs(left.astype(np.float32) - right.astype(np.float32))
    return {
        "surface": name,
        "shape": list(left.shape),
        "shape_match": left.shape == right.shape,
        "max_abs": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs": float(np.mean(diff)) if diff.size else 0.0,
        "off_finite": _finite_status(left),
        "experimental_finite": _finite_status(right),
    }


def _first_divergent_stage(
    off_entries: list[dict[str, Any]], exp_entries: list[dict[str, Any]]
) -> str | None:
    exp_by_key = {
        _trace_key(entry): np.asarray(entry.get("array"), dtype=np.float32)
        for entry in exp_entries
        if entry.get("array") is not None
    }
    for entry in off_entries:
        if entry.get("array") is None:
            continue
        key = _trace_key(entry)
        other = exp_by_key.get(key)
        if other is None:
            return str(entry.get("stage"))
        value = np.asarray(entry["array"], dtype=np.float32)
        if value.shape != other.shape or not np.array_equal(value, other):
            return str(entry.get("stage"))
    return None


def _trace_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        entry.get("case"),
        entry.get("layer"),
        entry.get("head"),
        entry.get("token_index"),
        entry.get("stage"),
    )


def _stack_trace_stage(entries: list[dict[str, Any]], stage: str) -> np.ndarray:
    arrays = [
        np.asarray(entry["array"], dtype=np.float32)
        for entry in entries
        if entry.get("stage") == stage and entry.get("array") is not None
    ]
    if not arrays:
        return np.zeros((0,), dtype=np.float32)
    return np.stack(arrays, axis=0)


def _summary(name: str, value: Any) -> dict[str, Any]:
    return summarize_array(name, value).__dict__


def _finite_status(value: Any) -> dict[str, int]:
    array = np.asarray(value)
    return {
        "finite_count": int(np.isfinite(array).sum()),
        "nonfinite_count": int((~np.isfinite(array)).sum()),
        "nan_count": int(np.isnan(array).sum()),
    }


def _mode_flags(config: RWKV7QwenReferenceConfig) -> dict[str, bool]:
    return {
        "radlads_balance_state_terms": config.radlads_balance_state_terms,
        "radlads_balance_state": config.radlads_balance_state,
        "use_radlads_balance_state_terms": config.use_radlads_balance_state_terms,
        "radlads_compatible_math": config.radlads_compatible_math,
    }


def _off_flags_are_default(flags: dict[str, bool]) -> bool:
    return not any(flags.values())


def _json_array(value: Any) -> Any:
    return np.asarray(value).tolist()


def _max_abs(left: Any, right: Any) -> float:
    diff = np.abs(
        np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)
    )
    return float(np.max(diff)) if diff.size else 0.0


def _without_arrays(payload: dict[str, Any]) -> dict[str, Any]:
    clean = dict(payload)
    clean["cases"] = [
        {
            key: value
            for key, value in case.items()
            if key not in {"arrays", "trace_entries"}
        }
        for case in payload["cases"]
    ]
    return clean


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_case_arrays(
    manifest_path: Path, case: dict[str, Any]
) -> dict[str, np.ndarray]:
    with np.load(manifest_path.parent / case["payload"]) as payload:
        return {name: payload[name] for name in payload.files}


def _prepare_out_dir(out_dir: Path, *, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise SystemExit(f"{out_dir} is not empty; pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)


def _experiment_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# P65 Balance-State Experiment",
        "",
        f"- Status: `{report['status']}`",
        f"- Default/off preserved: `{report['default_behavior_preserved']}`",
        f"- Off mode: `{report['off_mode_status']}`",
        f"- Experimental mode: `{report['experimental_mode_status']}`",
        "",
        "| Case | Status | First divergent stage |",
        "| --- | --- | --- |",
    ]
    for case in report["cases"]:
        lines.append(
            f"| {case['name']} | `{case['status']}` | "
            f"`{case['first_divergent_stage']}` |"
        )
    lines.extend(
        [
            "",
            "The experiment keeps `radlads_balance_state_terms` and "
            "`radlads_balance_state` as explicit opt-in switches. It does not "
            "change default behavior, add Pallas kernels, or claim full "
            "RADLADS parity.",
            "",
        ]
    )
    return "\n".join(lines)


def _diff_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# P65 Balance-State Diff Summary",
        "",
        "| Case | Surface | Shape | Max abs | Nonfinite off/experimental |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in report["cases"]:
        for item in case["comparisons"]:
            off_nonfinite = item["off_finite"]["nonfinite_count"]
            exp_nonfinite = item["experimental_finite"]["nonfinite_count"]
            lines.append(
                f"| {case['name']} | `{item['surface']}` | `{item['shape']}` | "
                f"`{item['max_abs']:.8g}` | `{off_nonfinite}/{exp_nonfinite}` |"
            )
    lines.append("")
    return "\n".join(lines)


def _off_vs_experimental_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# P65 Off vs Experimental",
        "",
        f"- Default/off preserved: `{report['default_behavior_preserved']}`",
        f"- Off mode status: `{report['off_mode_status']}`",
        f"- Experimental mode status: `{report['experimental_mode_status']}`",
        "",
    ]
    for case in report["cases"]:
        diverged = case["status"] != "same"
        lines.append(
            f"- {case['name']}: `{case['status']}`"
            + (
                f" (first divergent stage: `{case['first_divergent_stage']}`)"
                if diverged
                else ""
            )
        )
    lines.append("")
    return "\n".join(lines)


def _stability_markdown(report: dict[str, Any]) -> str:
    deltas = report["full_vs_stepwise"]
    return "\n".join(
        [
            "# P65 Balance-State Stability Smoke",
            "",
            f"- Status: `{report['status']}`",
            f"- Finite: `{report['finite']}`",
            f"- Hidden full-vs-step max abs: `{deltas['hidden_states_max_abs']}`",
            f"- Logits full-vs-step max abs: `{deltas['logits_max_abs']}`",
            f"- WKV state full-vs-step max abs: `{deltas['wkv_matrix_state_max_abs']}`",
            "",
        ]
    )
