# WKV7 Correctness Fixtures

P43 adds a tiny deterministic fixture harness for the extracted WKV7 matrix
recurrence/state core. The harness is intentionally narrower than the full
student block: it tests projected per-head `r`, `w`, `k`, `v`, `a`, `b`, and
`gate` arrays, an initial matrix state, optional attention masks, full-scan
outputs, and next matrix state.

The canonical artifact directory is `artifacts/kernels/p43_wkv7_correctness`.
It contains `manifest.json`, `P43_WKV7_FIXTURE_SUMMARY.md`,
`comparison_report.json`, `P43_WKV7_COMPARISON_REPORT.md`, and
`cases/*/{inputs,expected}.npz`.

The manifest records `schema_version: 0.1`, `phase: P43`,
`fixture_set: tiny_wkv7_correctness`, source git metadata, per-case shapes,
dtypes, paths, hashes, tolerances, and explicit stepwise-vs-full-scan metrics.
The current fixture set covers six deterministic tiny cases spanning no-mask,
masked, prefix-padding/reset-style, non-zero initial state, stateful
step-by-step equivalence, and extreme-but-finite decay values.

Generate fixtures with:

```bash
python scripts/generate_wkv7_correctness_fixtures.py \
  --out artifacts/kernels/p43_wkv7_correctness \
  --overwrite
```

Compare a candidate with:

```bash
python scripts/compare_wkv7_correctness_fixtures.py \
  --manifest artifacts/kernels/p43_wkv7_correctness/manifest.json \
  --candidate reference \
  --out artifacts/kernels/p43_wkv7_correctness \
  --overwrite
```

`reference` recomputes the current slow JAX recurrence and should pass. `pallas`
is a clean unsupported placeholder in P43; it reports `unsupported` until a real
optimized kernel exists.

Statuses are explicit: `pass`, `fail`, `unsupported`, `missing_fixture`,
`shape_mismatch`, `dtype_mismatch`, `non_finite`, and `candidate_error`.
Reports include output and next-state shape, dtype, finite status,
`max_abs_error`, `mean_abs_error`, `max_relative_error`, and `allclose` where
the candidate produces comparable arrays.

Tolerance policy:

- `float32`: `atol=1e-5`, `rtol=1e-5`
- `bfloat16`: `atol=5e-2`, `rtol=5e-2` for future surfaced bf16 candidates

P43 does not implement or benchmark a production Pallas WKV7 kernel. It also
does not optimize TPU performance, test the full Qwen/RADLADS block, claim
checkpoint parity, or make training/evaluation quality claims.

Relationship to P40: P40 measured source-surface comparability at the broader
RADLADS/Qwen bridge, while P43 narrows the scope to the extracted WKV7
recurrence/state core so future optimized kernels have a deterministic local
gate before any performance work lands.
