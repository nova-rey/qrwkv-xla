# P86 Pallas Fused/Scan Sequence Scaffold Report

## Scope
- one_step_formula: `state * decay[..., None, :] + k[..., :, None] * v[..., None, :]`
- sequence_parity_scope: `fused_or_scan_style_wkv_sequence`
- sequence_method: `jax_scan_pallas_step_scaffold`
- fused_sequence_kernel_status: `scan_scaffold_pass`
- default_runtime: `reference`
- Pallas opt-in: `True`

## Case Matrix
| case_id | batch | heads | dim | seq_len | dtype | sequence_method | fused_sequence_kernel_status | parity_status | final_max_abs_error | final_max_rel_error | worst_step_max_abs_error | worst_step_max_rel_error | atol | rtol |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| float32_b1_h1_d2_t2 | 1 | 1 | 2 | 2 | float32 | jax_scan_pallas_step_scaffold | scan_scaffold_pass | pass | 0.0 | 0.0 | 0.0 | 0.0 | 1e-06 | 1e-06 |
| float32_b1_h1_d2_t4 | 1 | 1 | 2 | 4 | float32 | jax_scan_pallas_step_scaffold | scan_scaffold_pass | pass | 7.450580596923828e-09 | 7.876759866576322e-08 | 7.450580596923828e-09 | 7.876759866576322e-08 | 1e-06 | 1e-06 |
| float32_b1_h2_d2_t4 | 1 | 2 | 2 | 4 | float32 | jax_scan_pallas_step_scaffold | scan_scaffold_pass | pass | 1.4901161193847656e-08 | 1.0148435336532202e-07 | 2.9802322387695312e-08 | 1.1450320158701288e-07 | 1e-06 | 1e-06 |
| float32_b2_h2_d4_t4 | 2 | 2 | 4 | 4 | float32 | jax_scan_pallas_step_scaffold | scan_scaffold_pass | pass | 1.1920928955078125e-07 | 2.211218514958091e-07 | 2.384185791015625e-07 | 2.2789797071709472e-07 | 1e-06 | 1e-06 |
| bfloat16_b1_h1_d2_t4 | 1 | 1 | 2 | 4 | bfloat16 | jax_scan_pallas_step_scaffold | scan_scaffold_pass | pass | 0.0 | 0.0 | 0.0 | 0.0 | 0.05 | 0.05 |
| bfloat16_b2_h2_d4_t4 | 2 | 2 | 4 | 4 | bfloat16 | jax_scan_pallas_step_scaffold | scan_scaffold_pass | pass | 0.0 | 0.0 | 0.0 | 0.0 | 0.05 | 0.05 |

## Summary
- cases_total: `6`
- cases_pass: `6`
- cases_fail: `0`
- cases_unavailable: `0`
- all_required_cases_pass: `True`
- kernel_parity_claimed: `True`
- fused_sequence_kernel_status: `scan_scaffold_pass`
- worst_final_max_abs_error: `1.1920928955078125e-07`
- worst_final_max_rel_error: `2.211218514958091e-07`
- worst_step_max_abs_error: `2.384185791015625e-07`
- worst_step_max_rel_error: `2.2789797071709472e-07`

## P85 Preservation
- p85_repeated_step_matrix_preserved: `True`
- p85_kernel_parity_claimed: `True`

## Capture Semantics
- pallas_requested_reference_trace_contamination: `False`
- reference_trace_capture_skipped: `True`

## Decision
- recommended_next_phase: `P87 fixture-family opt-in Pallas runtime integration`
