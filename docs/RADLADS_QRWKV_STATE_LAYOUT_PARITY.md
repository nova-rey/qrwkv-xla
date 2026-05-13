# RADLADS-QRWKV State/Layout Parity Diagnostics

P55 is the diagnostic phase after P54 made the clean payload loader/export path
real.

## What P54 proved

- RADLADS repo not modified
- clean deterministic payload loads
- RADLADS and QRWKV outputs both export
- logits pass across all tiny cases
- shift_state passes across all tiny cases

## What P55 diagnoses

- hidden_states convention mismatch
- wkv_matrix_state layout or pre/post-update mismatch
- stepwise surface coverage and status classification

## Conventions

- RADLADS hidden_states: final hidden
- QRWKV hidden_states: layer-major all hidden
- wkv_matrix_state: compare the exported recurrent state directly
- stepwise surfaces: only compare when the source actually provides them

## Candidate analysis

P55 runs non-mutating candidate transforms before any source-backed code change.
That keeps layout/convention issues separate from true recurrence math.

## What P55 does not do

- no Pallas
- no TPU optimization
- no training claim
- no model-quality claim
- no tolerance loosening to hide the residual

## Kernel readiness

Only after hidden_states and wkv_matrix_state are explicit and credible at the
slow-reference level.
