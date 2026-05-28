# P83 Pallas Reference Parity Report

## Runtime Selector
- default runtime: `reference`
- allowed runtimes: `['reference', 'pallas']`
- reference default preserved: `True`
- CLI/config path: `--wkv-runtime reference|pallas`, `wkv_runtime`

## Tiny Reference-vs-Pallas Parity Gate
- pallas requested: `True`
- pallas available: `True`
- pallas effective runtime: `pallas`
- prototype_status: `pass`
- parity_status: `pass`
- parity_scope: `tiny_one_step_wkv_update`
- probe_backend: `pallas_call_interpret`
- probe_shapes: `{'state': [1, 1, 2, 2], 'k': [1, 1, 2], 'v': [1, 1, 2], 'decay': [1, 1, 2], 'output': [1, 1, 2, 2]}`
- shape_match: `True`
- finite: `True`
- max_abs_error: `0.0`
- max_rel_error: `0.0`
- atol: `1e-06`
- rtol: `1e-06`
- kernel_parity_claimed: `True`

## Capture Semantics
- pallas_requested_reference_trace_contamination: `False`
- fail_closed_before_capture: `False`
- reference_trace_capture_skipped: `True`

## Scope
- formula: `state * decay[..., None, :] + k[..., :, None] * v[..., None, :]`
- full Pallas kernel readiness: `not_claimed_by_p83`
- default runtime promotion: `not_performed`

## Decision
- recommended_next_phase: `P86 fused/scan Pallas WKV kernel scaffold`
