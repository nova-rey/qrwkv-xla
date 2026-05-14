# RADLADS WKV State Export Convention

## P60 finding summary
P60 narrowed the remaining real-artifact blocker to the WKV matrix-state surface, but the real paired outputs still showed a residual after the `log_w` fix:

- `logits`: preserved
- `shift_state`: preserved
- `wkv_matrix_state`: still failing
- `hidden_states`: comparison/shape-convention noise downstream
- `kernel_ready`: no

Current real-artifact audit (`artifacts/p61_wkv_state_export_convention/`) shows the same pattern: the exported WKV matrix state slot names line up, but the raw comparison still does not reach tolerance.

## RADLADS state tuple structure
RADLADS exports a per-layer recurrent cache that behaves like:

- slot 0: `wkv_matrix_state`
- slot 1: `shift_state`

The cached artifact path is the HF/RADLADS `past_key_values` export path, later flattened into `radlads_wkv_matrix_state` and `radlads_shift_state` in the clean loader.

## QRWKV state tuple structure
QRWKV exports an explicit state object:

- slot 0: `wkv_matrix_state`
- slot 1: `shift_state`
- slot 2: `next_position`

The returned state is the `RWKV7QwenReferenceState` tuple from `apply_with_state` / `step`.

## RADLADS exported wkv_matrix_state source path
- `src/qrwkv_xla/students/rwkv7_radlads_reference.py`
- `src/qrwkv_xla/parity/radlads_clean_loader.py`
- `src/qrwkv_xla/parity/radlads_head_to_head.py`

RADLADS exports `wkv_matrix_state` from the recurrent cache after the full sequence finishes.

## QRWKV exported wkv_matrix_state source path
- `src/qrwkv_xla/students/rwkv7_qwen_reference.py`
- `src/qrwkv_xla/parity/radlads_numerical_fixtures.py`
- `src/qrwkv_xla/parity/radlads_wkv_state_provenance.py`

QRWKV exports `wkv_matrix_state` from the returned `RWKV7QwenReferenceState` after the sequence finishes.

## internal WKV state semantics on each side
Both sides use the same conceptual object: a per-layer, per-batch, per-head matrix cache updated token by token during recurrence.

## returned/cached WKV state semantics on each side
- RADLADS: returned through cache/past-key-values semantics
- QRWKV: returned through explicit `RWKV7QwenReferenceState`

## diagnostic/exported WKV state semantics on each side
Both sides export the post-update state for diagnostics. The comparison surface is the exported final cache, not an internal pre-update scratch buffer.

## pre-update vs post-update convention
Current audit favors a post-update exported state on both sides. No source-backed evidence showed that the remaining residual is caused by a pre/post mismatch.

## full-sequence vs stepwise convention
The slot audit reports the same slot names in full-sequence and stepwise paths. No source-backed slot swap was needed for the real-artifact audit.

## mask/padding export convention
Masked-token handling is already surfaced separately through `shift_state` and hidden-state diagnostics. The WKV residual persists even when mask behavior is held constant.

## comparison normalization before P61
- as-is comparison
- no source-backed slot swap
- no axis rewrite
- no tolerance loosening

## comparison normalization after P61
The P61 inspection script still recommends `as_is` for the real artifacts. The named slots match, but the residual remains:

- raw `wkv_matrix_state` error: `0.00036038574762642384`
- normalized `wkv_matrix_state` error: `0.00036038574762642384`

So P61 is diagnostic-only for now.

## fix applied, if any
No source-backed fix was applied in this phase.

## remaining caveats
- The WKV residual is still unresolved on the real cached artifacts.
- Hidden-state mismatch remains a comparison/shape-convention issue, not a kernel proof.
- Pallas remains blocked until the remaining residual is explained or removed.

## kernel readiness
`kernel_ready: no`
