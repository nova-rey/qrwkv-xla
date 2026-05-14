# RADLADS WKV Composite Balance Hook

## P62/P63 context
P62 narrowed the remaining WKV residual to the update path around `decayed_state`, `update_term`, and `state_after`.
P63 completed the live hook scaffolding but still reported the composite/balance-state hook as unavailable in captured traces.

## Why the composite/balance-state hook matters
The missing term is the balance-state matmul addend inside the full WKV state update. If it diverges, it can explain the remaining `wkv_matrix_state` residual.

## RADLADS source expression
`next_state = prev_state * decay[:, :, None, :] + prev_state @ ab + vk`

## QRWKV-XLA source expression
`next_wkv = prev_wkv * decay[:, :, None, :] + prev_wkv @ ab + vk`

## Whether each side materializes the term
Both sides materialize the balance-state contribution inline as `prev_state @ ab` / `prev_wkv @ ab`.

## Whether each side can capture the term live
P63 traces do not expose it live, but the term can be reconstructed exactly from `state_after - decayed_state - update_outer_product`.

## Whether exact reconstruction is possible
Yes, from source-backed ingredients already captured in the live trace.

## Capture strategy
- live hook when available
- exact reconstruction from `state_after`, `decayed_state`, and `update_outer_product` when not
- partial reconstruction only if a required ingredient is missing

## Comparison result
The P64 extraction/comparison path reconstructs the term exactly on both sides and compares the reconstructed tensors directly.

## Whether the term explains the residual
It does not appear to be the remaining culprit once reconstructed on both sides; the residual closes after the full source formula is rebuilt.

## Recommended next phase
P65 residual-impact / kernel-readiness gate

## Source table

| side | source file | function/class | source expression | source variable name | comparison label | capture method | reason if unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RADLADS | `src/qrwkv_xla/students/rwkv7_radlads_reference.py` | `rwkv7_radlads_reference_layer/step` | `next_state = prev_state * decay[:, :, None, :] + prev_state @ ab + vk` | `ab` | `composite_balance_update_term` | `exact_reconstruction` | live hook not present in P63 trace |
| QRWKV-XLA | `src/qrwkv_xla/students/rwkv7_qwen_reference.py` | `RWKV7QwenReference.step/apply_with_state` | `next_wkv = prev_wkv * decay[:, :, None, :] + prev_wkv @ ab + vk` | `ab` | `composite_balance_update_term` | `exact_reconstruction` | live hook not present in P63 trace |
