# P85 Pallas Sequence Parity Report

## Scope
- one_step_formula: `state * decay[..., None, :] + k[..., :, None] * v[..., None, :]`
- sequence_parity_scope: `short_sequence_repeated_one_step_wkv`
- default_runtime: `reference`
- Pallas opt-in: `True`
- sequence_method: `repeated_one_step_pallas`

## Case Matrix
| case_id | batch | heads | dim | seq_len | dtype | parity_status | final_max_abs_error | final_max_rel_error | worst_step_max_abs_error | worst_step_max_rel_error | atol | rtol |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| float32_b1_h1_d2_t2 | 1 | 1 | 2 | 2 | float32 | pass | 0.0 | 0.0 | 0.0 | 0.0 | 1e-06 | 1e-06 |
| float32_b1_h1_d2_t4 | 1 | 1 | 2 | 4 | float32 | pass | 7.450580596923828e-09 | 7.876759866576322e-08 | 7.450580596923828e-09 | 7.876759866576322e-08 | 1e-06 | 1e-06 |
| float32_b1_h2_d2_t4 | 1 | 2 | 2 | 4 | float32 | pass | 1.4901161193847656e-08 | 1.0148435336532202e-07 | 2.9802322387695312e-08 | 1.1450320158701288e-07 | 1e-06 | 1e-06 |
| float32_b2_h2_d4_t4 | 2 | 2 | 4 | 4 | float32 | pass | 1.1920928955078125e-07 | 2.211218514958091e-07 | 2.384185791015625e-07 | 2.2789797071709472e-07 | 1e-06 | 1e-06 |
| bfloat16_b1_h1_d2_t4 | 1 | 1 | 2 | 4 | bfloat16 | pass | 0.0 | 0.0 | 0.0 | 0.0 | 0.05 | 0.05 |
| bfloat16_b2_h2_d4_t4 | 2 | 2 | 4 | 4 | bfloat16 | pass | 0.0 | 0.0 | 0.0 | 0.0 | 0.05 | 0.05 |

## Summary
- cases_total: `6`
- cases_pass: `6`
- cases_fail: `0`
- cases_unavailable: `0`
- all_required_cases_pass: `True`
- kernel_parity_claimed: `True`
- worst_final_max_abs_error: `1.1920928955078125e-07`
- worst_final_max_rel_error: `2.211218514958091e-07`
- worst_step_max_abs_error: `2.384185791015625e-07`
- worst_step_max_rel_error: `2.2789797071709472e-07`

## Capture Semantics
- pallas_requested_reference_trace_contamination: `False`
- reference_trace_capture_skipped: `True`

## Decision
- recommended_next_phase: `P86 fused/scan Pallas WKV kernel scaffold`
