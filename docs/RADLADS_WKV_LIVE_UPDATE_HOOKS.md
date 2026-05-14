# RADLADS WKV Live Update Hooks

## P62 context
P62 narrowed the remaining residual to the WKV update path but still lacked a source-backed composite live hook.

## Why P63 exists
P63 exists to expose or explicitly label the live recurrence substages around `decayed_state`, `update_outer_product`, `balance_state_matmul`, `composite_update_term`, `update_term`, and `state_after`.

## RADLADS live WKV update source path
- source file: `src/qrwkv_xla/students/rwkv7_radlads_reference.py`
- source function: `RWKV7RadladsReference.step/apply_with_state`

## QRWKV live WKV update source path
- source file: `src/qrwkv_xla/students/rwkv7_qwen_reference.py`
- source function: `RWKV7QwenReference.step/apply_with_state`

## Complete update formula on each side
The comparison target remains the same conceptual recurrence:
- `decayed_state`
- `update_outer_product`
- `balance_state_matmul / composite_update_term`
- `update_term`
- `state_after`

## Which intermediate rows are now exposed
The live hook trace exposes the named rows where they exist and explicitly labels missing composite hooks as unavailable or reconstructed.

## Which rows remain unavailable, if any
`balance_state_matmul` and `composite_update_term` remain unavailable in the real tiny traces unless reconstructed for comparison-only reporting.

## First live substage divergence
The current live hook comparison still points at the earliest missing composite hook surface rather than a proven recurrence math mismatch.

## Source-backed root cause hypothesis
The residual is still most likely a missing composite live hook or a comparison-normalization gap, not a proven recurrence rewrite target.

## Whether a recurrence fix is now safe
No. P63 is still instrumentation/comparison-first.

## Kernel readiness
`kernel_ready: no`
