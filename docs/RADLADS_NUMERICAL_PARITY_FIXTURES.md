# RADLADS Tiny Numerical Parity Fixtures

P49 adds a bounded fixture path for real tiny RADLADS numerical comparisons.
It now generates real local RADLADS tiny fixtures in this environment, but the
current comparison result is still intentionally narrow: parameter-surface
comparison is live, while QRWKV-XLA output/state value replay remains marked
unsupported until the remaining runtime-critical parameter import gaps are
closed. P49 does not claim full RADLADS parity, checkpoint import, training,
Pallas kernels, a Hugging Face model class, or Qwen-scale execution.

## Scope

The P49 schema is for tiny deterministic cases only:

- `tiny_no_mask`
- `tiny_attention_mask`
- `tiny_prefix_or_left_padding`
- `tiny_stepwise_state`
- `tiny_all_radlads_math_enabled`

Each case records input arrays, optional attention mask metadata, surface names,
array shapes and dtypes, payload hashes, per-case status, and declared
comparisons. Comparison surfaces may include parameters, block outputs, hidden
states, recurrent state, logits, and stepwise/full-state behavior when real
RADLADS source arrays are available.

## Status Vocabulary

Per-case statuses are:

- `pass`
- `fail`
- `unsupported`
- `missing_source`
- `fail_known_difference`

Overall report statuses are:

- `pass`
- `pass_with_known_differences`
- `fail`
- `source_unavailable`

The manifest also records `real_radlads_fixture_status` as one of:

- `generated`
- `imported`
- `source_unavailable`
- `execution_failed`

Default tolerances are float32 `atol=1e-5` and `rtol=1e-5`. The schema records
a looser bfloat16 policy of `atol=3e-2` and `rtol=3e-2` for source fixtures that
surface bfloat16 arrays.

## Offline-Safe Import

The default importer writes QRWKV-XLA current-behavior payloads and marks every
case `missing_source`. These payloads are useful for schema, hashing, and report
coverage, but they are not RADLADS outputs.

```bash
python scripts/import_radlads_tiny_numerical_fixtures.py \
  --out artifacts/p49_radlads_numerical_parity/radlads_fixtures \
  --overwrite
```

To import real source-produced fixtures:

```bash
python scripts/import_radlads_tiny_numerical_fixtures.py \
  --source-dir /path/to/p49/radlads_tiny_numerical \
  --out artifacts/p49_radlads_numerical_parity/radlads_fixtures \
  --overwrite
```

## Optional Live Generation

Live RADLADS execution is disabled by default. To attempt it, set:

```bash
QRWKV_XLA_RUN_RADLADS_LIVE_FIXTURES=1 \
python scripts/generate_radlads_tiny_numerical_fixtures.py \
  --radlads-source /home/nyx/.openclaw/workspace/_refs/RADLADS \
  --out artifacts/p49_radlads_numerical_parity/radlads_fixtures \
  --overwrite
```

The generator uses the actual local RADLADS source path and patches in a tiny
CPU-only fallback recurrent kernel plus a minimal cache shim so the source can
emit deterministic arrays without Triton/CUDA. If source execution is still
unavailable or fails, it writes an honest `execution_failed` or
`source_unavailable` manifest rather than fabricating RADLADS arrays from
QRWKV-XLA.

## Comparison Reports

Run:

```bash
python scripts/compare_radlads_tiny_numerical_fixtures.py \
  --manifest artifacts/p49_radlads_numerical_parity/radlads_fixtures/manifest.json \
  --out artifacts/p49_radlads_numerical_parity
```

Outputs:

- `numerical_parity_report.json`
- `P49_RESULTS.md`
- `surface_comparison.json`
- `P49_SURFACE_COMPARISON.md`

## Parameter Mapping

`qrwkv_xla.parity.radlads_parameter_mapping` provides minimal shape/name
mapping statuses for tiny numerical fixtures:

- `mapped_exact`
- `mapped_renamed`
- `shape_mismatch`
- `missing_in_qrwkv`
- `missing_in_radlads`
- `unsupported`
- `source_not_found`

This is not a checkpoint importer. It is enough to make tiny fixture
comparisons explicit and auditable, and to show exactly which surfaces are
already shape-compatible versus still missing or intentionally unsupported.

## Current Caveat

In the current P49 result, real RADLADS fixtures are generated successfully and
parameter surfaces compare live, but hidden/logits/state parity against
QRWKV-XLA is still reported as `unsupported` because the remaining value-import
bridge for runtime-critical surfaces is incomplete. That is deliberate: P49 is
honest about what is and is not comparable today.
