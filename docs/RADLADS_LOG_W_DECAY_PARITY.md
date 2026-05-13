# RADLADS log_w Decay Parity

P57 is a source-audit / caliper phase for the first divergent P56 WKV trace
stage: `log_w`.

It adds:

- `src/qrwkv_xla/parity/radlads_log_w_parity.py`
- `scripts/compare_radlads_qrwkv_log_w.py`
- `tests/test_radlads_log_w_decay_parity.py`

The comparison script reads RADLADS `log_w` rows from a JSONL trace artifact,
captures QRWKV `log_w` from the current model path via diagnostics, compares
the rows, and evaluates non-mutating candidate formula variants across:
orientation, sign, activation, base-term, dtype, and axis handling.

Default command:

```bash
python scripts/compare_radlads_qrwkv_log_w.py --overwrite
```

Default artifacts are written to `artifacts/p57_log_w_decay_parity/`:

- `log_w_parity_report.json`
- `P57_LOG_W_PARITY.md`
- `log_w_values.npz`
- `P57_LOG_W_CANDIDATES.md`
- `log_w_candidate_report.json`

P57 does not patch model math. If current QRWKV `log_w` matches RADLADS on the
tiny trace, the report is diagnostic-only and records that. If it mismatches,
the mismatch is source-backed by the captured RADLADS trace and current QRWKV
diagnostics, but still remains diagnostic until a separate phase decides on a
model change.

Observed results from the current run:

- `first_mismatch`: `tiny_no_mask / L0 / H0 / T0`
- `max_abs_error`: `0.003779083490371704`
- candidate best: `as_is__negative__sigmoid__exp_neg_half__float64__as_is`
  (still failing, max abs error `0.0037790801710292365`)
- trace rerun first divergent stage: `log_w`
- trace rerun `wkv_state_after` growth at token 3: `2.2600870579481125e-06`
- head-to-head summary: `attempted_comparisons=40`, `pass=12`, `fail=12`,
  `not_applicable=16`
- best passing surface: `tiny_no_mask:logits`
- largest remaining failure: `hidden_states` (max abs error `0.060687560588121414`)
- `wkv_matrix_state` remains finite but mismatched (max abs error
  `0.00036038458347320557`)

No tolerances are loosened. The default comparison remains `atol=1e-5` and
`rtol=1e-5`, matching the surrounding tiny parity diagnostics.
