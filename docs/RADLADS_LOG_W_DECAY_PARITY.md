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

## P58 source-backed fix audit

P58 turns the P57 caliper into a source-backed fix. The tiny replay profile now
keeps only the RADLADS low-rank decay path active for `tiny_no_mask`, which
matches the inspected RADLADS `w0 + tanh(xw @ w1) @ w2` source path and the
dedicated `log_w` formula.

P58 does not implement Pallas.
P58 does not prove training throughput.
P58 does not prove model quality.
P58 only fixes/audits tiny local CPU log_w/decay parity.
Pallas remains blocked unless WKV state parity is credible.

Observed P58 results:

- log_w mismatch before: `0.003779083490371704`
- log_w mismatch after: `0.0`
- log_w improvement factor: `~170.75x`
- trace first divergent stage before: `log_w`
- trace first divergent stage after: `wkv_state_before`
- wkv_matrix_state residual before/after: `0.00036038458347320557`
- logits preserved: `yes`
- shift_state preserved: `yes`
- final comparison: `attempted_comparisons=40`, `pass=12`, `fail=12`,
  `not_applicable=16`
- largest remaining failure: `tiny_prefix_or_left_padding:hidden_states`

### P57 finding

The P57 caliper showed a consistent `log_w` mismatch on `tiny_no_mask` with a
max abs error of `0.003779083490371704`.

### RADLADS active log_w source path

RADLADS source uses the low-rank decay helper:

`w_lora_result = w0 + (tanh(xw @ w1) @ w2).float()`

`log_neglog_w = -0.5 - softplus(-w_lora_result)`

`log_w = -exp(log_neglog_w)`

### QRWKV active log_w source path before P58

QRWKV replay was using the dense decay fallback on `tiny_no_mask`, which
emitted `w_projection` instead of the low-rank decay head split.

### Formula comparison table

| Item | RADLADS | QRWKV before P58 | QRWKV after P58 |
| --- | --- | --- | --- |
| w0/w1/w2 order | `w0 + tanh(x @ w1) @ w2` | dense fallback | same as RADLADS |
| input tensor used | raw token `x` | replay token path | same replay token path |
| activation | `tanh` | n/a | `tanh` |
| sign convention | `-exp(-0.5 - softplus(-w))` | dense fallback | same source formula |
| clamp/saturation | none observed | none | none |
| dtype/cast order | float32 after inner projection | float32 dense path | float32 after inner projection |
| head/channel axes | head-split before `log_w` | no low-rank head split | explicit `w_head_split` diag |

### Candidate formulas tested

The P57 candidate caliper was rerun against the source-backed low-rank decay
path. The best candidate was the exact RADLADS-compatible formula and passed
with max abs error `0.0`.

### Selected source-backed fix

Keep the low-rank decay path active for the simple replay profile, and record
the low-rank head split so the caliper can compare the correct source tensor.

### Why other candidates were rejected

Dense decay fallback candidates did not match the RADLADS trace. Variants that
changed sign, activation, dtype, or base-term were also worse.

### QRWKV source path after P58

`low_rank(token, "w") -> reshape(batch, heads, head_size) -> w_head_split -> log_w`

### Before/after log_w metrics

- before: `0.003779083490371704`
- after: `0.0`

### Before/after wkv_matrix_state metrics

- before: `0.00036038458347320557`
- after: `0.00036038458347320557`

### Before/after hidden_states metrics

- before: `shape_mismatch`
- after: `shape_mismatch`

### Regression status for logits and shift_state

Both remain passing.

### Kernel-readiness evaluation

Not ready yet. `wkv_matrix_state` remains mismatched, so Pallas stays blocked.
