# RADLADS Source Parity Fixture Bridge

P40 adds a canonical fixture bridge for comparing QRWKV-XLA against
source-generated RADLADS arrays when those arrays are available. The bridge is
schema and reporting infrastructure, not a new parity claim.

## Fixture Schema

Canonical manifests use:

- `fixture_version: 1`
- `schema: radlads_source_parity.v1`
- `backend: rwkv7_qwen_reference`
- `cases`: one entry each for `tiny_no_mask`, `tiny_attention_mask`, and
  `tiny_prefix_padding_or_left_padding`

Each case points at an NPZ payload and records `input_ids`, optional
`attention_mask` metadata, payload hash, status, and comparison specs. Supported
comparison payloads should include source arrays such as
`radlads_hidden_states` plus QRWKV arrays such as `qrwkv_hidden_states`.

The checked-in default fixtures under `tests/fixtures/radlads_source_parity`
are intentionally marked `unsupported`. They contain deterministic QRWKV-XLA
current-behavior arrays only. They are useful for schema, loader, and report
coverage, but they are not RADLADS outputs and must not be counted as numerical
parity.

## Import Path

Use the import script to validate and copy real source-generated canonical
fixtures:

```bash
./.venv/bin/python scripts/import_radlads_source_fixtures.py \
  --source-fixtures /path/to/canonical/radlads/fixtures \
  --out tests/fixtures/radlads_source_parity \
  --overwrite
```

When `--source-fixtures` is omitted, the script regenerates the deterministic
QRWKV current-behavior-only fixtures and keeps every case unsupported. This is
the default offline CI-safe path.

P40 does not add live RADLADS execution as a normal test path. The local source
checkout at `/home/nyx/.openclaw/workspace/_refs/RADLADS` depends on PyTorch,
Transformers, FLA/Triton-oriented runtime paths, and source-specific setup. A
future live generator should remain explicitly env-gated and should write the
same canonical schema rather than replacing it.

## Comparison Reports

Run:

```bash
./.venv/bin/python scripts/compare_radlads_source_fixtures.py \
  --manifest tests/fixtures/radlads_source_parity/manifest.json \
  --out-dir artifacts/parity/radlads_source_bridge
```

The script writes:

- `artifacts/parity/radlads_source_bridge/parity_report.json`
- `artifacts/parity/radlads_source_bridge/P40_PARITY_REPORT.md`

Case statuses are:

- `pass`: all declared comparisons are present, shape-compatible, and within
  tolerance.
- `fail`: at least one declared comparison is missing, shape-incompatible, or
  outside tolerance.
- `unsupported`: the case has no fair source comparison, usually because
  RADLADS arrays are not present or the parameter surface is not equivalent.

## Parameter Surface Map

Run:

```bash
./.venv/bin/python scripts/map_radlads_parameter_surface.py \
  --out-dir artifacts/parity/radlads_source_bridge
```

The script writes:

- `artifacts/parity/radlads_source_bridge/parameter_surface_map.json`
- `artifacts/parity/radlads_source_bridge/P40_PARAMETER_SURFACE_MAP.md`

The map records direct role matches such as embeddings, Q/K/V/O projections,
RMSNorm, MLP, final norm, and LM head. It marks low-rank RADLADS paths
(`w0/w1/w2`, `a0/a1/a2`, `v0/v1/v2`, gate variants, `k_k/k_a/r_k`, and optional
`ln_x`) as unsupported for numerical parity against the current QRWKV-XLA
parameterization.

## Scope Boundary

This bridge preserves offline behavior. It does not add real training, TPU
training, optimized WKV kernels, Pallas, `pjit`, sharding, Qwen-scale export,
HF student export, `lm_eval`, WandB, or full RADLADS numerical parity claims.
