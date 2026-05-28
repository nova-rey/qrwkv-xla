# P84 Pallas Shape/Dtype Parity Report

## Scope
- formula: `state * decay[..., None, :] + k[..., :, None] * v[..., None, :]`
- parity_scope: `broader_one_step_wkv_shape_dtype`
- default runtime: `reference`
- Pallas opt-in: `True`

## Case Matrix
| case_id | batch | heads | dim | dtype | parity_status | finite | shape_match | max_abs_error | max_rel_error | atol | rtol |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| float32_b1_h1_d2 | 1 | 1 | 2 | float32 | pass | True | True | 0.0 | 0.0 | 1e-06 | 1e-06 |
| float32_b1_h2_d2 | 1 | 2 | 2 | float32 | pass | True | True | 0.0 | 0.0 | 1e-06 | 1e-06 |
| float32_b2_h1_d2 | 2 | 1 | 2 | float32 | pass | True | True | 0.0 | 0.0 | 1e-06 | 1e-06 |
| float32_b1_h1_d4 | 1 | 1 | 4 | float32 | pass | True | True | 0.0 | 0.0 | 1e-06 | 1e-06 |
| float32_b2_h2_d4 | 2 | 2 | 4 | float32 | pass | True | True | 0.0 | 0.0 | 1e-06 | 1e-06 |
| bfloat16_b1_h1_d2 | 1 | 1 | 2 | bfloat16 | pass | True | True | 0.0 | 0.0 | 0.05 | 0.05 |
| bfloat16_b2_h2_d4 | 2 | 2 | 4 | bfloat16 | pass | True | True | 0.0 | 0.0 | 0.05 | 0.05 |

## Summary
- cases_total: `7`
- cases_pass: `7`
- cases_fail: `0`
- cases_unavailable: `0`
- all_required_cases_pass: `True`
- kernel_parity_claimed: `True`

## Capture Semantics
- pallas_requested_reference_trace_contamination: `False`
- reference_trace_capture_skipped: `True`

## Decision
- recommended_next_phase: `P85 sequence/scan-style Pallas WKV parity`
