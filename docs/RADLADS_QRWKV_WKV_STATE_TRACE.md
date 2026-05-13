# RADLADS-QRWKV WKV State Trace

P56 adds a trace-first view of the remaining WKV residual after P55 ruled out the obvious layout/export explanation.

## What it does
- captures per-token WKV stage traces for RADLADS and QRWKV on tiny cases
- compares `log_w`, `wkv_state_before`, `wkv_state_after`, and output stages
- reports the first divergent WKV stage and candidate update-order matches

## Interpretation
- `log_w` divergence points at decay/update convention or dtype-order sensitivity
- `wkv_state_after` is the key recurrence surface before the final projection
- if `as_is` remains best, the residual is not a simple axis swap

## How to run
- `python scripts/trace_radlads_qrwkv_wkv_update.py --manifest artifacts/p54_confirmation/fixtures/manifest.json --radlads-outputs artifacts/p54_confirmation/radlads_outputs/manifest.json --qrwkv-outputs artifacts/p54_confirmation/qrwkv_outputs/manifest.json --case tiny_no_mask --out artifacts/p56_wkv_state_residual_trace --overwrite`
- `python scripts/compare_radlads_qrwkv_wkv_trace.py --radlads-trace artifacts/p56_wkv_state_residual_trace/wkv_trace_radlads.jsonl --qrwkv-trace artifacts/p56_wkv_state_residual_trace/wkv_trace_qrwkv.jsonl --out artifacts/p56_wkv_state_residual_trace/trace_comparison --overwrite`
- `python scripts/analyze_wkv_update_order_candidates.py --radlads-trace artifacts/p56_wkv_state_residual_trace/wkv_trace_radlads.jsonl --qrwkv-trace artifacts/p56_wkv_state_residual_trace/wkv_trace_qrwkv.jsonl --out artifacts/p56_wkv_state_residual_trace/trace_comparison --overwrite`

## Caveats
- P56 does not implement Pallas.
- P56 does not prove training throughput.
- P56 does not prove model quality.
- P56 only diagnoses tiny local CPU WKV state parity.
- Pallas remains blocked unless the WKV matrix-state residual is explained and credible.
