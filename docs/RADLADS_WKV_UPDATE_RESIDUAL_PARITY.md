# RADLADS WKV Update Residual Parity

## P58/P60/P61 context
- P58 fixed the `log_w` / decay formula exactly.
- P60 traced the remaining real paired-artifact residual and left it in the state/export territory.
- P61 ruled out the obvious WKV matrix-state slot/export convention mismatch.

## Why export/slot convention is no longer the lead suspect
P61 compared the named slots and export surfaces, and the raw vs normalized WKV matrix-state error stayed unchanged. That leaves the update path itself as the next reasonable target.

## RADLADS WKV update formula/source path
- `src/qrwkv_xla/students/rwkv7_radlads_reference.py`
- `src/qrwkv_xla/parity/radlads_wkv_trace.py`

## QRWKV WKV update formula/source path
- `src/qrwkv_xla/students/rwkv7_qwen_reference.py`
- `src/qrwkv_xla/parity/radlads_wkv_trace.py`

## state_before comparison
The first residual trace still shows `state_before` aligned on the captured tiny traces.

## decay_applied comparison
The residual appears before a clean `decayed_state` comparison can be completed on the QRWKV side for the traced artifact set, so the exact recurrence substage still needs a live source hook.

## update_term comparison
`update_outer_product` is captured on the RADLADS side, but the composite `update_term` surface is not separately captured in the existing trace rows.

## state_after comparison
The first divergence shows up at `decayed_state` / `state_after` surfaces in the current comparison report, with the first measured error remaining small but nonzero.

## dtype/cast comparison
Both traced sides report `float64` in the comparison artifact set; no dtype-only fix is proven.

## mask/padding interaction
Masked-case behavior is still diagnostic-only. The update trace does not expose a separate mask gate on the update term.

## full vs stepwise interaction
The current residual trace remains unresolved under the existing real-artifact full/stepwise surfaces.

## candidate fixes tested
- export/slot normalization: rejected in P61
- as-is comparison: still fails on the residual surface
- no numeric tolerance loosening

## fix applied, if any
No source-backed recurrence fix was applied.

## remaining residual
The current P62 comparison report still fails with `kernel_ready: no`.

## kernel readiness decision
`kernel_ready: no`
