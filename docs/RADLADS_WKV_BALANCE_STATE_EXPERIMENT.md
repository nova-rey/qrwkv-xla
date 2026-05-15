# RADLADS WKV Balance-State Experiment

P65 adds a small opt-in experiment for the existing QRWKV-XLA balance-state
compatibility path. The experiment treats
`RWKV7QwenReferenceConfig.radlads_balance_state_terms` plus
`radlads_balance_state` as the explicit experimental switch.

## Scope

- Default/off behavior remains unchanged.
- Experimental mode is enabled only when both
  `radlads_balance_state_terms=True` and `radlads_balance_state=True`.
- The existing source-backed balance-state path in
  `src/qrwkv_xla/students/rwkv7_qwen_reference.py` is reused.
- P58 `log_w` behavior and P63/P64 update-hook behavior are preserved.
- No Pallas, TPU optimization, broad recurrence rewrite, tolerance loosening,
  or default promotion is included.

## Scripts

Run the fixture comparison:

```bash
python scripts/run_balance_state_experiment.py --overwrite
```

Run the tiny local stability smoke:

```bash
python scripts/run_balance_state_stability_smoke.py --overwrite
```

Both scripts are CPU/local and deterministic. They use the checked-in tiny
fixture manifest by default and instantiate the slow JAX reference student
directly.

## Artifacts

Default output directory:
`artifacts/p65_balance_state_experiment/`

The experiment writes:

- `P65_RESULTS.md`
- `balance_state_experiment_report.json`
- `DIFF_SUMMARY.md`
- `OFF_VS_EXPERIMENTAL.md`
- `off/mode_report.json`
- `off/mode_arrays.json`
- `experimental/mode_report.json`
- `experimental/mode_arrays.json`

The stability smoke writes:

- `STABILITY_SMOKE.md`
- `stability/stability_report.json`

## Compared Surfaces

The experiment compares off vs experimental mode for:

- `log_w`
- `logits`
- `hidden_states`
- `wkv_matrix_state`
- `shift_state`
- first divergent diagnostic stage
- finite, NaN, and nonfinite counts
- state/output summary statistics
- explicit mode and flag status

## Interpretation

P65 is an experiment/stability surface, not a parity claim. A diff between off
and experimental mode is expected because the experimental mode wires the
existing source-backed balance-state update path behind the flag. The contract
is that off/default mode remains bit-for-bit unchanged and the experimental
path is visible, deterministic, finite, and locally smoke-tested.

## P66 caveat

P66 does not implement Pallas.
P66 does not prove training throughput.
P66 does not prove model quality.
P66 does not promote experimental balance_state mode by default.
P66 only measures whether experimental balance_state mode moves QRWKV-XLA
closer to RADLADS on tiny/local parity fixtures.

## P67 caveat

P67 does not implement Pallas.
P67 does not prove training throughput.
P67 does not prove model quality.
P67 does not promote experimental balance_state mode by default.
P67 only establishes a same-run ingredient-level RADLADS-vs-QRWKV comparison
on tiny/local fixtures.
