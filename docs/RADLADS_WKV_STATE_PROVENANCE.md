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
