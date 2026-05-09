# RADLADS Replay Non-Finite Diagnostics

P50 proved that QRWKV-XLA could import the real P49 RADLADS tiny parameter payload, but its replay pass forced the all-math RADLADS path on every fixture. That was wrong for four of the five real P49 cases: `tiny_no_mask`, `tiny_attention_mask`, `tiny_prefix_or_left_padding`, and `tiny_stepwise_state` were generated with `all_radlads_math=False`.

That mismatch activated low-rank replay surfaces that were inactive in the source fixture path. The shared parameter payload also contains suspicious/non-finite low-rank source tensors, so the forced all-math profile turned simple finite fixtures into replay-side `non_finite` failures.

## What P51 adds

- Tensor-summary diagnostics for replay tensors and final outputs.
- First-nonfinite detection in replay order.
- Parameter sanity reporting for the RADLADS payload.
- Case-specific replay profiles so simple P49 fixtures no longer force the all-math RADLADS path.
- Explicit reporting of QRWKV-only defaulted surfaces that are still active in the chosen replay profile.

## Run diagnostics

```bash
python scripts/diagnose_radlads_replay_nonfinite.py \
 --manifest artifacts/p49_radlads_numerical_parity/radlads_fixtures/manifest.json \
 --parameters artifacts/p49_radlads_numerical_parity/radlads_fixtures/radlads_parameters.npz \
 --case tiny_no_mask \
 --out artifacts/p51_radlads_replay_diagnostics \
 --overwrite
```

Outputs:

- `artifacts/p51_radlads_replay_diagnostics/P51_DIAGNOSTIC_REPORT.md`
- `artifacts/p51_radlads_replay_diagnostics/replay_diagnostics.json`
- `artifacts/p51_radlads_replay_diagnostics/tensor_summaries.jsonl`
- `artifacts/p51_radlads_replay_diagnostics/P51_PARAMETER_SANITY.md`
- `artifacts/p51_radlads_replay_diagnostics/parameter_sanity_report.json`

## Re-run stabilized replay

```bash
python scripts/replay_radlads_tiny_numerical_fixtures.py \
 --manifest artifacts/p49_radlads_numerical_parity/radlads_fixtures/manifest.json \
 --parameters artifacts/p49_radlads_numerical_parity/radlads_fixtures/radlads_parameters.npz \
 --out artifacts/p51_radlads_replay_stabilized \
 --report-prefix P51 \
 --overwrite
```

Outputs:

- `artifacts/p51_radlads_replay_stabilized/P51_RESULTS.md`
- `artifacts/p51_radlads_replay_stabilized/replay_comparison_report.json`
- `artifacts/p51_radlads_replay_stabilized/P51_SURFACE_COMPARISON.md`

## How to read the reports

- `first_nonfinite: null` with `final_outputs_finite: true` means replay stayed finite and comparisons should now fall through to numeric errors or passes.
- A non-null `first_nonfinite` shows the earliest replay tensor that blew up, with stage, layer, optional time index, counts, and first bad location.
- Parameter sanity reports are the first stop when replay looks bad. If source parameters are already non-finite or absurdly large, replay math is not the first suspect.
- `qrwkv_only_default_used_in_active_path` highlights replay profiles that still rely on defaulted QRWKV-only leaves such as `time_mix`, `time_bias`, or `b_proj.weight`.

## What changed in P51

- Replay profiles now follow the original fixture intent instead of enabling all RADLADS math globally.
- The simple P49 fixtures can produce finite QRWKV-XLA outputs again.
- The all-math fixture remains diagnostic-only until the suspicious source low-rank payload is resolved.

## Caveats

- P51 does not implement Pallas.
- P51 does not prove full RADLADS parity unless compared surfaces actually pass.
- P51 does not prove training throughput.
- P51 does not prove model quality.
- The all-math fixture can still fail because the inspected source payload contains suspicious/non-finite low-rank tensors.

Pallas stays blocked until the slow replay reference is finite for the relevant fixture profile.
