# P87 Pallas Fixture-Family Integration Report

- phase: `P87`
- status: `pass`
- runtime_default_preserved: `True`
- pallas_opt_in_preserved: `True`
- reference_contamination_detected: `False`
- fixture_alias_behavior_preserved: `True`
- cases_total: `6`
- cases_passed: `6`
- cases_failed: `0`
- cases_skipped: `0`
- unsupported_cases: `[]`
- parity_scope: `covered_fixture_family_opt_in_pallas_runtime`
- recommended_next_phase: `P88 TPU compile/performance smoke`

## Case Matrix
| case_id | status | output_max_abs_error | next_state_max_abs_error | reason |
| --- | --- | ---: | ---: | --- |
| tiny_b1_t4_h1_d4_no_mask | pass | 0.0 | 0.0 |  |
| tiny_b2_t5_h2_d4_no_mask | pass | 0.0 | 0.0 |  |
| tiny_b1_t6_h2_d4_with_attention_mask | pass | 0.0 | 0.0 |  |
| tiny_prefix_padding_or_reset_case | pass | 0.0 | 0.0 |  |
| tiny_stateful_step_vs_full_scan | pass | 0.0 | 0.0 |  |
| tiny_extreme_but_finite_decay_values | pass | 0.0 | 0.0 |  |

## Claims Not Made
- production Pallas readiness
- training readiness
- TPU readiness
- throughput proof
- Pallas default readiness
- full model parity
