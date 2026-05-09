# RADLADS Parameter Replay Compatibility

P50 adds a bounded replay path for the real tiny RADLADS parameter payload
produced by P49.

## What P50 Proves

- QRWKV-XLA can load the real tiny `radlads_parameters.npz` payload.
- QRWKV-XLA can build an explicit RADLADS replay config for the slow reference.
- QRWKV-XLA can import mapped parameter values, report defaulted/excluded/
  unsupported surfaces honestly, and attempt real replay on the P49 fixture
  inputs.
- QRWKV-XLA can produce measured pass/fail/unsupported/missing_source/
  shape_mismatch/dtype_mismatch/non_finite statuses for replay surfaces.

## What P50 Does Not Prove

- P50 does not prove full RADLADS checkpoint compatibility.
- P50 does not implement Pallas.
- P50 does not prove training throughput.
- P50 attempts tiny replay compatibility only.
- P50 does not prove model quality.

## Why P49 Replay Was Unsupported

P49 generated real local RADLADS tiny fixtures, but replay stopped at parameter
surface comparison because QRWKV-XLA still lacked:

- q/k/v projection bias import
- replay-mode handling for QRWKV-only legacy surfaces
- a parameter importer that could consume the real RADLADS payload
- a replay comparator that could run the same tiny cases through the slow
  reference and measure output/state differences

P50 fills that gap.

## q/k/v Bias Handling

Replay mode enables q/k/v bias support through the slow reference config flag:

- `attention_qkv_bias: true`

The importer maps and loads:

- `layers.self_attn.q_proj.bias`
- `layers.self_attn.k_proj.bias`
- `layers.self_attn.v_proj.bias`

Legacy no-bias behavior remains available when replay mode / bias mode is not
enabled.

## g1/g2 Handling

The local source inspected for P50 is:

- `/home/nyx/.openclaw/workspace/_refs/RADLADS/rwkv7qwen2/modeling_rwkv7qwen2.py`

For `gate_rank_type == 2`, the inspected source computes the low-rank gate as:

```text
sigmoid(xg @ g1) @ g2
```

P50 implements that expression in explicit replay mode. This means `g1/g2` are
not left as unsupported placeholders in the current bounded replay path.

## QRWKV-only Surface Handling

These QRWKV-only surfaces are not required from the RADLADS payload:

- `a_proj.weight`
- `b_proj.weight`
- `g_proj.weight`
- `w_proj.weight`
- `time_mix`
- `time_bias`
- `lm_head.bias`

In replay mode they are never silently random-initialized and claimed as parity.
Instead they are explicitly reported as either:

- `defaulted` (for deterministic zero/default behavior)
- `excluded` (when source-backed replay says the surface is not active)
- `unsupported` (if a surface still falls outside the bounded replay map)

`r_k` is explicitly excluded because the inspected RADLADS source keeps the
residual use commented out in the active path.

## How to Run Replay Comparison

```bash
python scripts/replay_radlads_tiny_numerical_fixtures.py \
  --manifest artifacts/p49_radlads_numerical_parity/radlads_fixtures/manifest.json \
  --parameters artifacts/p49_radlads_numerical_parity/radlads_fixtures/radlads_parameters.npz \
  --out artifacts/p50_radlads_replay_compatibility \
  --overwrite
```

Expected outputs:

- `artifacts/p50_radlads_replay_compatibility/P50_RESULTS.md`
- `artifacts/p50_radlads_replay_compatibility/replay_comparison_report.json`
- `artifacts/p50_radlads_replay_compatibility/parameter_import_report.json`
- `artifacts/p50_radlads_replay_compatibility/P50_PARAMETER_IMPORT_REPORT.md`
- `artifacts/p50_radlads_replay_compatibility/P50_SURFACE_COMPARISON.md`

## How to Read Statuses

Surface comparison statuses include:

- `pass`
- `fail`
- `unsupported`
- `missing_source`
- `shape_mismatch`
- `dtype_mismatch`
- `non_finite`
- `not_replayed_due_to_import_failure`

Unsupported surfaces do not count as passes.

## Why This Comes Before Pallas

Pallas kernels would only make mismatches harder to understand right now. P50
keeps the work on the slow reference so parameter mapping, replay semantics, and
numerical differences stay auditable before any optimized kernel path is added.
