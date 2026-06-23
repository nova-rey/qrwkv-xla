# P156.1 Recovery Report

## Recovery Status

- Recovery performed without rerunning pytest, experiments, or training.
- No pytest, P156, P156.1, or quality-per-byte process remained active when inspected.
- The prior full pytest process had exited and its terminal session was no longer available.
- The last recoverable pytest progress was 98%. A final pytest footer and exit code were not recoverable, so final suite status is **unknown**.
- Before the terminal session disappeared, exactly two failures had been displayed. Both were the pre-existing macOS x86/JAX Qwen reference-fixture hash failures described below. No P156.1 failure appeared.

## Commands Recovered

Focused validation completed before recovery:

```bash
ruff check .
git diff --check
uv run --no-project --with 'jax[cpu]<0.5' --with pytest --with-editable . pytest -q tests/test_budgeted_artifact.py tests/test_controlled_quality_per_byte.py tests/test_fingerprint_quality_per_byte_experiment.py tests/test_two_cycle_experiment.py
```

Confirmed result: `51 passed, 1 skipped in 96.54s`.

Later focused validation completed before recovery:

```bash
uv run --no-project --with 'jax[cpu]<0.5' --with pytest --with-editable . pytest -q tests/test_budgeted_artifact.py tests/test_controlled_quality_per_byte.py tests/test_two_cycle_experiment.py::test_unconfounded_byte_family_fast_cpu_smoke tests/test_two_cycle_experiment.py::test_controlled_quality_per_byte_fast_cpu_smoke tests/test_two_cycle_experiment.py::test_sequential_two_cycle_integration_smoke
```

Confirmed result: `23 passed in 13.78s`.

The full regression command was:

```bash
uv run --no-project --with 'jax[cpu]<0.5' --with pytest --with-editable . pytest -q
```

Last confirmed progress: 98%; final exit status unavailable.

The final CPU matrix command was an inline Python driver launched through:

```bash
uv run --no-project --with 'jax[cpu]<0.5' --with pytest --with-editable . python - <<'PY'
# Built three disjoint tiny fingerprint artifacts and one selected-profile receipt.
# Ran bytes family: byte_budgets=(1000, 2000), fixed_total_steps=4,
# seeds=(0, 1, 2), backend=cpu, student_backend=tiny_debug.
# Ran steps family: step_budgets=(2, 4), fixed_artifact_budget=2000,
# seeds=(0, 1, 2), backend=cpu, student_backend=tiny_debug.
PY
```

The exact inline source is not available as a file. Its complete effective configuration is preserved in the reports, matrix-state files, subset indices, and per-cell config hashes.

## Matrix Completion

### Final Byte-Controlled Matrix

- Root: `/tmp/p1561-final-4um77248/bytes`
- Completed: 24/24 arm cells, 0 failed, 0 pending.
- Shape: 2 actual byte ceilings x 3 aligned seeds x 4 arms.
- Fixed optimizer budget: 4 total steps per arm.
- Backend: CPU, single host.
- Actual budgeted subsets consumed: true.
- Byte-family integrity: valid.
- Quality-per-byte claim allowed: true, tiny CPU scope only.
- Result: `exemplar_only_better` at both byte ceilings.
- Classification: **scientifically valid tiny-CPU byte-controlled matrix**.

### Final Step-Controlled Matrix

- Root: `/tmp/p1561-final-4um77248/steps`
- Completed: 24/24 arm cells, 0 failed, 0 pending.
- Shape: 2 optimizer-step budgets x 3 aligned seeds x 4 arms.
- Fixed artifact ceiling: 2000 charged logical payload bytes.
- Subset hashes fixed across step points.
- Backend: CPU, single host.
- Step-family integrity: valid.
- Quality-per-step claim allowed: true, tiny CPU scope only.
- Result: `exemplar_only_better` at both step budgets.
- Classification: **scientifically valid tiny-CPU step-controlled matrix**.

### Time-Controlled Matrix

- True deadline enforcement was not implemented or run.
- Reports label this family `deferred` and set `quality_per_time_claim_allowed=false`.
- Classification: **not run; no scientific claim**.

## `exemplar_only_better` Classification

The final `exemplar_only_better` result came from the completed full local scientific matrices, not from a unit test:

- Byte-controlled matrix: two real ceilings, three aligned seeds, four arms, fixed steps.
- Step-controlled matrix: two step points, three aligned seeds, four arms, fixed artifact subsets.

The text also appears in earlier diagnostic P156/P156.1 outputs and per-cell P155 reports. Those occurrences are not the basis for the final classification.

## Test Failures

### Pre-existing/environment-specific

The full suite displayed two failures before reaching the last confirmed 98%:

1. `test_qwen_reference_fixture_generation_is_deterministic`
2. `test_qwen_reference_fixture_script_smoke`

Both regenerate Qwen numerical fixture hashes differently under constrained JAX 0.4 on macOS x86. They reproduced the same failures from the prior full-suite run, where the final result was `1360 passed, 24 skipped, 2 failed`. They are outside the P156.1 diff.

### P156.1 development diagnostics

- `/tmp/p1561-smoke-tbij7syk`: 0/4 cells; early aggregation/development failure.
- `/tmp/p1561-smoke2-qkvw84s7`: 0/4 cells; P155 subset-backed cell status failure during separate-artifact wiring.
- `/tmp/p1561-smoke3-dgpf0683`: 4/4 cells; one-point smoke passed but was non-publication-grade.
- `/tmp/p1561-required-0j2a57dv`: 48/48 cells across byte and step families; superseded because the final fixed initialization-subset control was added afterward.

These are diagnostic or superseded runs. The final matrices under `/tmp/p1561-final-4um77248` completed without failed cells.

## Output Inventory

Exact generated root sizes:

| Root | Files | Bytes | Classification |
|---|---:|---:|---|
| `/tmp/p156-required-bcamm6vd` | 610 | 1,376,566 | Diagnostic-only, confounded legacy P156 |
| `/tmp/p1561-final-4um77248` | 1,297 | 3,130,463 | Final scientifically valid byte/step matrices |
| `/tmp/p1561-required-0j2a57dv` | 1,277 | 3,056,285 | Complete but superseded diagnostic matrix |
| `/tmp/p1561-smoke-tbij7syk` | 131 | 179,219 | Failed development smoke |
| `/tmp/p1561-smoke2-qkvw84s7` | 157 | 249,690 | Failed development smoke |
| `/tmp/p1561-smoke3-dgpf0683` | 158 | 292,573 | Successful one-point diagnostic smoke |

The complete per-file inventory is split between:

- `docs/agent-handoffs/p156_1_preserved_outputs_index.json`: every preserved file with original path, repository path, exact size, SHA-256, and classification.
- `docs/agent-handoffs/p156_1_large_artifacts_manifest.json`: every omitted file with absolute path, exact size, SHA-256, and omission reason.

Preserved evidence includes quality-per-byte reports, byte/step/time reports, matrix states, subset indices, subset manifests, byte-accounting receipts, record-selection receipts, paired comparisons, quality curves, publication receipts, CPU receipts, summaries, and supplied specifications.

No dedicated file named `reuse_plan` was generated. Reuse semantics are preserved by `budget_subset_index.json`, subset cache keys, subset manifest hashes, and matrix config hashes.

No pytest log file was written. Test evidence existed only in terminal output; the recoverable summaries are recorded above.

## Scientific Classification

- `/tmp/p1561-final-4um77248/bytes`: scientifically valid for the narrow tiny-CPU byte-controlled claim.
- `/tmp/p1561-final-4um77248/steps`: scientifically valid for the narrow tiny-CPU step-controlled claim.
- Time-controlled outputs: deferred, claim-disabled.
- `/tmp/p156-required-bcamm6vd`: diagnostic-only because its byte/step/time views were confounded and actual byte ceilings were not causally enforced.
- `/tmp/p1561-required-0j2a57dv`: complete but superseded; diagnostic-only.
- All `smoke*` roots: diagnostic-only, incomplete, or failed as listed above.
- Full pytest final status: unknown beyond 98%; two known pre-existing failures, no observed P156.1 failures.
