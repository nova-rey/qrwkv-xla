# RADLADS WKV State Provenance

P59 adds a diagnostic provenance layer for WKV state handoff. It traces the
existing QRWKV student `apply_with_state` and `step` APIs and writes JSONL plus
JSON/Markdown reports for these surfaces:

- `initial_state`: explicit initial state versus a freshly initialized state.
- `initial_state_handoff`: explicit initial-state full run versus implicit
  initial-state full run.
- `token_carry`: each step input state versus the previous step output state.
- `full_vs_stepwise`: full-sequence output/state versus token-by-token
  output/state.
- `mask_behavior`: diagnostic deltas across masked token steps.

The module is `src/qrwkv_xla/parity/radlads_wkv_state_provenance.py`. The local
trace script is `scripts/trace_radlads_qrwkv_wkv_state_provenance.py`; by
default it writes under `artifacts/p59_wkv_state_provenance`. The comparison
script is `scripts/compare_radlads_qrwkv_wkv_state_provenance.py`.

P59 is diagnostic-only. It does not change recurrence math, widen tolerances,
claim full RADLADS parity, add Pallas kernels, or alter the P58 `log_w` fixes.
The synthetic test path uses `tmp_path` artifacts so CI does not require
ignored RADLADS artifacts.

Example local run:

```bash
python scripts/trace_radlads_qrwkv_wkv_state_provenance.py --overwrite
python scripts/compare_radlads_qrwkv_wkv_state_provenance.py \
  --radlads-trace artifacts/p59_wkv_state_provenance/wkv_state_provenance.jsonl \
  --qrwkv-trace artifacts/p59_wkv_state_provenance/wkv_state_provenance.jsonl \
  --out artifacts/p59_wkv_state_provenance/self_compare \
  --overwrite
```

Expected files:

- `wkv_state_provenance.jsonl`
- `wkv_state_provenance_report.json`
- `P59_WKV_STATE_PROVENANCE.md`
- `wkv_state_provenance_manifest.json`
- `comparison/wkv_state_provenance_report.json`
- `self_compare/wkv_state_provenance_comparison_report.json`
- `self_compare/wkv_state_provenance_report.json`
- `self_compare/P59_WKV_STATE_PROVENANCE_COMPARISON.md`
- `self_compare/P59_WKV_STATE_PROVENANCE.md`

## P60 Real Artifact Provenance

P60 adds a real-artifact layer on top of the P59 provenance schema. It consumes
paired cached deterministic tiny artifacts from `artifacts/p54_confirmation`
and the post-fix WKV traces from `artifacts/p58_log_w_decay_fix/post_fix_trace`.
It does not regenerate live RADLADS outputs in this environment and therefore
labels every emitted row with:

- `real_artifact_trace: true`
- `synthetic_trace: false`
- `self_comparison_trace: true`
- `derived_from_cached_outputs: true`
- `regenerated_live_outputs: false`

The P60 runner is
`scripts/run_real_radlads_qrwkv_wkv_state_provenance.py`; the comparator is
`scripts/compare_real_radlads_qrwkv_wkv_state_provenance.py`. Default outputs
live under `artifacts/p60_real_wkv_state_provenance/`.

Required P60 reports:

- `real_wkv_state_provenance_radlads.jsonl`
- `real_wkv_state_provenance_qrwkv.jsonl`
- `real_provenance_metadata.json`
- `p60_real_state_provenance_report.json`
- `P60_RESULTS.md`
- `TRACE_PROVENANCE.md`
- `comparison/P60_REAL_WKV_STATE_PROVENANCE.md`
- `comparison/p60_real_wkv_state_provenance_report.json`
- `TINY_NO_MASK_REAL_STATE.md`
- `TINY_STEPWISE_REAL_STATE.md`
- `REAL_MASK_PADDING_STATE.md`
- `HIDDEN_STATE_DEPENDENCY.md`

Observed P60 result: the top-level real provenance report writes successfully
from cached real artifacts with no synthetic fallback. The paired RADLADS-vs-
QRWKV comparison remains a real failure; the deterministic first divergence is
`tiny_attention_mask / initial_state_handoff / wkv_matrix_state` with max abs
error `0.0003433828242123127`.

`--strict-real-artifacts` intentionally fails in this implementation because no
live regenerated RADLADS output path is proven. This is by design: P60 labels
cached-derived outputs honestly rather than silently substituting synthetic
traces.

## P61 WKV Matrix-State Export Convention Audit

P61 audits the remaining real-artifact WKV matrix-state residual by checking
the exported slot, the returned state, and the comparison convention. It keeps
the P60 diagnosis intact, documents slot/export semantics, and avoids any
broad recurrence rewrite or tolerance widening.
