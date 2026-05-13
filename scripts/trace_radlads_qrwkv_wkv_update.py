#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from qrwkv_xla.parity.radlads_clean_loader import load_radlads_clean_payload
from qrwkv_xla.parity.radlads_numerical_fixtures import load_numerical_case_arrays
from qrwkv_xla.parity.radlads_parameter_import import (
    import_radlads_parameters_for_replay,
)
from qrwkv_xla.parity.radlads_replay import (
    replay_profile_for_case,
    student_for_replay_profile,
)
from qrwkv_xla.parity.radlads_wkv_trace import (
    WKV_TRACE_SCHEMA,
    WKVTraceCollector,
    compare_trace_entries,
    write_trace_comparison_reports,
)

RADLADS_ROOT = Path("/home/nyx/.openclaw/workspace/_refs/RADLADS")
DEFAULT_OUT = Path("artifacts/p56_wkv_state_residual_trace")
DEFAULT_CASE = "tiny_no_mask"


def _install_fla_shim() -> None:
    if "fla.ops.rwkv7.chunk" in sys.modules:
        return
    fla = sys.modules.setdefault("fla", types.ModuleType("fla"))
    ops = sys.modules.setdefault("fla.ops", types.ModuleType("fla.ops"))
    rwkv7 = sys.modules.setdefault("fla.ops.rwkv7", types.ModuleType("fla.ops.rwkv7"))
    chunk = types.ModuleType("fla.ops.rwkv7.chunk")
    fused = types.ModuleType("fla.ops.rwkv7.fused_recurrent")

    def _missing(*args, **kwargs):
        raise RuntimeError("fla shim should be patched before use")

    chunk.chunk_rwkv7 = _missing  # type: ignore[attr-defined]
    fused.fused_recurrent_rwkv7 = _missing  # type: ignore[attr-defined]
    sys.modules["fla.ops.rwkv7.chunk"] = chunk
    sys.modules["fla.ops.rwkv7.fused_recurrent"] = fused
    rwkv7.chunk = chunk  # type: ignore[attr-defined]
    rwkv7.fused_recurrent = fused  # type: ignore[attr-defined]
    ops.rwkv7 = rwkv7  # type: ignore[attr-defined]
    fla.ops = ops  # type: ignore[attr-defined]


@contextmanager
def _patched(module: Any, **replacements: Any):
    originals = {name: getattr(module, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(module, name, value)


def _select_cases(
    manifest: dict[str, Any], *, case: str | None, all_cases: bool
) -> list[dict[str, Any]]:
    cases = list(manifest["cases"])
    if all_cases:
        return cases
    selected = next(
        (item for item in cases if item["name"] == (case or DEFAULT_CASE)), None
    )
    if selected is None:
        raise SystemExit(f"case not found: {case or DEFAULT_CASE}")
    return [selected]


def _inline_manifest(case: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "name": case["name"],
        "payload": case["payload"],
        "description": case.get("description"),
        "attention_mask": case.get("attention_mask"),
        "all_radlads_math": case.get("all_radlads_math", False),
        "status": case.get("status"),
    }
    return payload


def _normalize_qrwkv_head_major_entries(
    entries: list[dict[str, Any]],
    *,
    num_heads: int,
    head_size: int,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        array = entry.get("array")
        if (
            entry.get("stage") in {"wkv_output_before_o_proj", "o_proj_output"}
            and array is not None
        ):
            values = np.asarray(array)
            if values.ndim == 2 and values.shape[1] == num_heads * head_size:
                reshaped = values.reshape(values.shape[0], num_heads, head_size)
                for head_index in range(num_heads):
                    head_entry = dict(entry)
                    head_entry["head"] = head_index
                    head_slice = reshaped[:, head_index]
                    head_entry["array"] = head_slice.tolist()
                    head_entry["shape"] = [int(dim) for dim in head_slice.shape]
                    normalized.append(head_entry)
                continue
        normalized.append(entry)
    return normalized


def _record_output_traces(
    collector: WKVTraceCollector, *, side: str, case: str, output: Any, state: Any
) -> None:
    hidden = output.hidden_states
    logits = getattr(output, "logits", None)
    if hidden is not None:
        collector.record(
            f"{side}.final_hidden",
            hidden,
            stage="layer_output",
            layer=None,
            token_index=None,
        )
    if logits is not None:
        collector.record(
            f"{side}.logits",
            logits,
            stage="logits",
            layer=None,
            token_index=None,
        )
    if state is not None:
        collector.record(
            f"{side}.returned_wkv_matrix_state",
            state.wkv_matrix_state,
            stage="returned_wkv_matrix_state",
            layer=None,
            token_index=None,
        )
        collector.record(
            f"{side}.returned_shift_state",
            state.shift_state,
            stage="returned_shift_state",
            layer=None,
            token_index=None,
        )


def _trace_qrwkv_case(
    *,
    case: dict[str, Any],
    manifest_path: Path,
    qrwkv_config: Any,
    params: dict[str, Any],
    max_inline_values: int,
) -> tuple[WKVTraceCollector, dict[str, Any]]:
    arrays = load_numerical_case_arrays(manifest_path, case)
    profile = replay_profile_for_case(case)
    student = student_for_replay_profile(qrwkv_config, profile)
    collector = WKVTraceCollector(
        case=case["name"],
        side="qrwkv",
        include_arrays=True,
        max_inline_values=max_inline_values,
    )
    input_ids = np.asarray(arrays["input_ids"], dtype=np.int32)
    attention_mask = None
    if "attention_mask" in arrays:
        attention_mask = np.asarray(arrays["attention_mask"], dtype=np.int32)
    output, state = student.apply_with_state(  # type: ignore[call-arg]
        params,
        input_ids,
        attention_mask,
        diagnostics=collector,
    )
    collector.entries = _normalize_qrwkv_head_major_entries(
        collector.entries,
        num_heads=int(qrwkv_config.num_heads),
        head_size=int(qrwkv_config.head_size),
    )
    _record_output_traces(
        collector, side="qrwkv", case=case["name"], output=output, state=state
    )
    return collector, {
        "case": case["name"],
        "profile": profile.reason,
        "hidden_shape": list(np.asarray(output.hidden_states).shape),
        "logits_shape": None
        if output.logits is None
        else list(np.asarray(output.logits).shape),
    }


def _trace_radlads_case(
    *,
    case: dict[str, Any],
    manifest_path: Path,
    load_result: Any,
    max_inline_values: int,
) -> tuple[WKVTraceCollector, dict[str, Any]]:
    arrays = load_numerical_case_arrays(manifest_path, case)
    collector = WKVTraceCollector(
        case=case["name"],
        side="radlads",
        include_arrays=True,
        max_inline_values=max_inline_values,
    )
    input_ids = np.asarray(arrays["input_ids"], dtype=np.int64)
    attention_mask = None
    if "attention_mask" in arrays:
        attention_mask = np.asarray(arrays["attention_mask"], dtype=np.int64)

    torch = importlib.import_module("torch")
    rad_module = importlib.import_module("rwkv7qwen2.modeling_rwkv7qwen2")
    call_index = {"layer": 0}

    def _trace_step(
        r, log_w, k, v, kk_neg, kk_a, *, initial_state=None, output_final_state=False
    ):
        del output_final_state
        layer = call_index["layer"]
        call_index["layer"] += 1
        device = r.device
        if initial_state is None:
            state = torch.zeros(
                (r.shape[0], r.shape[2], r.shape[3], r.shape[3]),
                dtype=torch.float32,
                device=device,
            )
        else:
            state = initial_state.to(dtype=torch.float32)
        outputs = []
        for token_index in range(r.shape[1]):
            r_t = r[:, token_index]
            log_w_t = log_w[:, token_index]
            k_t = k[:, token_index]
            v_t = v[:, token_index]
            kk_neg_t = kk_neg[:, token_index]
            kk_a_t = kk_a[:, token_index]
            decay = torch.exp(log_w_t)
            vk = torch.einsum("bhi,bhj->bhij", v_t, k_t)
            ab = torch.einsum("bhi,bhj->bhij", kk_neg_t, kk_a_t)
            state_before = state
            state_decay = state_before * decay[:, :, None, :]
            state_after = state_decay + state_before @ ab.float() + vk.float()
            output_before_o_proj = torch.einsum("bhij,bhj->bhi", state_after, r_t)
            collector.record(
                "radlads.receptance_or_r",
                r_t,
                stage="receptance_or_r",
                layer=layer,
                token_index=token_index,
            )
            collector.record(
                "radlads.log_w",
                log_w_t,
                stage="log_w",
                layer=layer,
                token_index=token_index,
            )
            collector.record(
                "radlads.decay_after_transform",
                decay,
                stage="decay_after_transform",
                layer=layer,
                token_index=token_index,
            )
            collector.record(
                "radlads.wkv_state_before",
                state_before,
                stage="wkv_state_before",
                layer=layer,
                token_index=token_index,
            )
            collector.record(
                "radlads.wkv_update_outer_or_term",
                vk,
                stage="wkv_update_outer_or_term",
                layer=layer,
                token_index=token_index,
            )
            collector.record(
                "radlads.wkv_decay_applied",
                state_decay,
                stage="wkv_decay_applied",
                layer=layer,
                token_index=token_index,
            )
            collector.record(
                "radlads.wkv_state_after",
                state_after,
                stage="wkv_state_after",
                layer=layer,
                token_index=token_index,
            )
            collector.record(
                "radlads.wkv_output_before_o_proj",
                output_before_o_proj,
                stage="wkv_output_before_o_proj",
                layer=layer,
                token_index=token_index,
            )
            state = state_after
            outputs.append(output_before_o_proj.reshape(r.shape[0], -1))
        output = torch.stack(outputs, dim=1)
        return output, state

    with _patched(
        rad_module,
        fused_recurrent_rwkv7=_trace_step,
        chunk_rwkv7=_trace_step,
    ):
        model = load_result.model
        assert model is not None
        model.eval()
        with torch.no_grad():
            output = model(
                input_ids=torch.tensor(input_ids, dtype=torch.long),
                attention_mask=None
                if attention_mask is None
                else torch.tensor(attention_mask, dtype=torch.long),
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )

    hidden = getattr(output, "hidden_states", None)
    if hidden is not None:
        collector.record(
            "radlads.final_hidden",
            hidden,
            stage="layer_output",
            layer=None,
            token_index=None,
        )
    logits = getattr(output, "logits", None)
    if logits is not None:
        collector.record(
            "radlads.logits",
            logits,
            stage="logits",
            layer=None,
            token_index=None,
        )
    return collector, {
        "case": case["name"],
        "hidden_shape": None if hidden is None else list(np.asarray(hidden).shape),
        "logits_shape": None if logits is None else list(np.asarray(logits).shape),
    }


def _write_manifest(
    out_dir: Path,
    *,
    commit: str,
    manifest_path: Path,
    radlads_outputs: Path | None,
    qrwkv_outputs: Path | None,
    cases: list[dict[str, Any]],
    radlads_trace: Path,
    qrwkv_trace: Path,
    trace_comparison: Path,
    selected_layer: str,
    selected_head: str,
    selected_token: str,
    max_inline_values: int,
) -> None:
    payload = {
        "schema": WKV_TRACE_SCHEMA,
        "commit": commit,
        "source_manifest": str(manifest_path),
        "radlads_outputs": None if radlads_outputs is None else str(radlads_outputs),
        "qrwkv_outputs": None if qrwkv_outputs is None else str(qrwkv_outputs),
        "cases": cases,
        "traces": {
            "radlads": str(radlads_trace),
            "qrwkv": str(qrwkv_trace),
        },
        "comparison": str(trace_comparison),
        "filters": {
            "layer": selected_layer,
            "head": selected_head,
            "token": selected_token,
            "max_inline_values": max_inline_values,
        },
    }
    (out_dir / "wkv_trace_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Trace RADLADS/QRWKV WKV recurrence stages for P56."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--radlads-outputs", type=Path)
    parser.add_argument("--qrwkv-outputs", type=Path)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--layer", default="all")
    parser.add_argument("--head", default="all")
    parser.add_argument("--token", default="all")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-inline-values", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.out} is not empty; pass --overwrite")
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = _select_cases(manifest, case=args.case, all_cases=args.all_cases)
    param_payload = args.manifest.parent / str(
        manifest.get("parameter_payload", "radlads_parameters.npz")
    )
    _install_fla_shim()
    load_result = load_radlads_clean_payload(
        param_payload,
        radlads_source_path=RADLADS_ROOT,
        seed=int(manifest.get("seed", 5353)),
        run_smoke=False,
    )
    if load_result.model is None:
        raise SystemExit(f"RADLADS model did not load: {load_result.reason}")
    import_result = import_radlads_parameters_for_replay(
        param_payload,
        allow_defaults=True,
        seed=int(manifest.get("seed", 5353)),
    )
    params = import_result.params
    qrwkv_config = load_result.qrwkv_config
    if qrwkv_config is None:
        raise SystemExit("QRWKV config missing from RADLADS clean payload loader")

    rad_entries: list[dict[str, Any]] = []
    qrw_entries: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    for case in cases:
        rad_collector, rad_summary = _trace_radlads_case(
            case=case,
            manifest_path=args.manifest,
            load_result=load_result,
            max_inline_values=args.max_inline_values,
        )
        qrw_collector, qrw_summary = _trace_qrwkv_case(
            case=case,
            manifest_path=args.manifest,
            qrwkv_config=qrwkv_config,
            params=params,
            max_inline_values=args.max_inline_values,
        )
        rad_entries.extend(rad_collector.entries)
        qrw_entries.extend(qrw_collector.entries)
        case_summaries.append(
            {"case": case["name"], "radlads": rad_summary, "qrwkv": qrw_summary}
        )

    rad_path = args.out / "wkv_trace_radlads.jsonl"
    qrw_path = args.out / "wkv_trace_qrwkv.jsonl"
    rad_collector = WKVTraceCollector(case=DEFAULT_CASE, side="radlads")
    qrw_collector = WKVTraceCollector(case=DEFAULT_CASE, side="qrwkv")
    rad_collector.extend(rad_entries)
    qrw_collector.extend(qrw_entries)
    rad_collector.write_jsonl(rad_path)
    qrw_collector.write_jsonl(qrw_path)

    trace_report = compare_trace_entries(rad_entries, qrw_entries)
    trace_out = args.out / "trace_comparison"
    write_trace_comparison_reports(trace_report, trace_out)
    _write_manifest(
        args.out,
        commit=manifest.get("commit", "unknown"),
        manifest_path=args.manifest,
        radlads_outputs=args.radlads_outputs,
        qrwkv_outputs=args.qrwkv_outputs,
        cases=case_summaries,
        radlads_trace=rad_path,
        qrwkv_trace=qrw_path,
        trace_comparison=trace_out,
        selected_layer=str(args.layer),
        selected_head=str(args.head),
        selected_token=str(args.token),
        max_inline_values=args.max_inline_values,
    )
    print(f"wrote WKV traces to {args.out}")
    print(f"first divergent stage={trace_report.get('first_divergent_stage')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
