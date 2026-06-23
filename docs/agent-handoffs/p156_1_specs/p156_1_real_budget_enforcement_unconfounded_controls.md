# P156.1 — Real Budget Enforcement and Unconfounded Efficiency Controls

## Status

Planned

## Purpose

P156 implemented a substantial controlled quality-per-byte orchestration scaffold:

```text
CPU backend enforcement
resumable budget × seed matrix
four-arm P155 reuse
paired bootstrap comparisons
quality curves
resource receipts
claim gates
```

Two scientific-control gaps prevent the current matrix from supporting a real quality-per-byte result:

1. Declared artifact-byte budgets are not causally enforced on the records consumed by each training arm.
2. Byte, optimizer-step, and wall-clock budgets vary together inside one bundled budget point, confounding the three efficiency views.

P156.1 must repair those two issues without redesigning the underlying training methods.

This is a narrow accounting, subset-materialization, and experimental-control phase.

Do not change:

```text
corridor loss
exemplar loss
selected hammer
P155 arm definitions
P155.1 split integrity
student architecture
primary final-test metric
```

---

# 1. Required Outcome

P156.1 must turn the existing P156 scaffold into three independently controlled experiment families:

```text
A. byte-controlled family
B. step-controlled family
C. time-controlled family
```

Each family must vary exactly one primary resource while holding the others fixed or non-binding.

The primary publication-grade result for P156 remains:

```text
byte-controlled corridor → exemplar vs exemplar-only
```

---

# 2. Deterministic Artifact Subset Materialization

Declared byte budgets must correspond to real materialized subsets consumed by the training runners.

Required module:

```text
src/qrwkv_xla/fingerprint/budgeted_artifact.py
```

or repository-equivalent location.

Required operation:

```text
full training artifact
+ deterministic selection policy
+ byte ceiling
+ arm payload role
→ budget-limited artifact subset
```

Each subset must be a valid standalone fingerprint artifact accepted by existing loaders.

Required subset roles:

```text
corridor_subset
exemplar_subset
combined_two_cycle_subset
```

The two-cycle arm may use separate corridor and exemplar subset manifests, but their total charged byte budget must not exceed the declared combined ceiling.

---

# 3. Byte Accounting Contract

P156.1 must distinguish:

```text
physical bytes on disk
logical payload bytes selected
logical payload bytes consumed
shared metadata bytes
arm-charged bytes
```

## 3.1 Physical Bytes

Actual serialized files belonging to the subset.

## 3.2 Logical Payload Bytes

Serialized or canonical encoded bytes of selected corridor/exemplar records, excluding reusable common metadata.

## 3.3 Shared Metadata

Tokenizer contracts, teacher identity, schema, and other common files may be stored once.

They must be reported separately.

Do not charge the same common metadata repeatedly across every arm unless the declared accounting policy explicitly does so.

## 3.4 Charged Bytes

The primary quality-per-byte denominator must be predeclared.

Recommended:

```text
arm_charged_logical_payload_bytes
```

Required report fields:

```text
byte_accounting_policy
declared_byte_budget
physical_subset_bytes
logical_payload_bytes_selected
logical_payload_bytes_consumed
shared_metadata_bytes
arm_charged_bytes
unused_budget_bytes
budget_ceiling_respected
```

Required invariant:

```text
arm_charged_bytes <= declared_byte_budget
```

Fail closed on violation.

---

# 4. Deterministic Record Selection

Subset selection must be deterministic from:

```text
source artifact hash
arm role
budget value
selection seed
selection policy version
```

Required ordering policy:

```text
stable canonical record order
```

or an explicitly seeded deterministic permutation.

Required receipt:

```text
budget_subset_manifest.json
```

Minimum:

```json
{
  "source_artifact_sha256": "...",
  "subset_role": "corridor_subset",
  "declared_byte_budget": 1000000,
  "selection_policy": "canonical_prefix_v1",
  "selection_seed": 0,
  "ordered_record_ids_sha256": "...",
  "selected_record_count": 1234,
  "logical_payload_bytes_selected": 998742,
  "physical_subset_bytes": 1012844,
  "budget_ceiling_respected": true,
  "subset_manifest_sha256": "..."
}
```

Selection must never use final-test scores.

---

# 5. Payload-Specific Subsets

## 5.1 Corridor-Only Arm

Its subset must contain only the corridor payload required by corridor training.

Do not charge unused exemplar payloads.

## 5.2 Exemplar-Only Arm

Its subset must contain only exemplar payloads and required source inputs.

Do not charge unused corridor payloads unless the loader contract genuinely requires them; if required, report them separately.

## 5.3 Two-Cycle Arm

The total declared byte budget must be partitioned:

```text
corridor allocation
+
exemplar allocation
=
total two-cycle byte ceiling
```

Required config:

```text
corridor_byte_fraction
```

or explicit:

```text
corridor_byte_budget
exemplar_byte_budget
```

The allocation must be frozen before final-test scoring.

Required:

```text
actual corridor charged bytes <= corridor allocation
actual exemplar charged bytes <= exemplar allocation
sum actual charged bytes <= total budget
```

---

# 6. Training Must Consume the Subsets

The existing P155 runners must receive the materialized subset paths.

Required evidence per arm:

```text
configured_training_artifact_path
configured_subset_manifest_sha256
loader_observed_artifact_sha256
records_available
records_consumed
logical_bytes_available
logical_bytes_consumed
```

The original full training artifact must not be passed into a byte-controlled cell except as the immutable source from which subsets are materialized.

Required failure:

```text
full_artifact_used_in_byte_controlled_cell
```

---

# 7. Independent Experiment Families

Replace bundled `QualityBudgetPoint(bytes, steps, wall_clock)` semantics with explicit family types.

Recommended schema:

```text
ByteControlledPoint
StepControlledPoint
TimeControlledPoint
```

or a single point with:

```text
control_family
varied_resource
fixed_resources
```

---

# 8. Byte-Controlled Family

This is the primary P156 experiment.

Vary:

```text
actual teacher-artifact byte ceiling
```

Hold fixed:

```text
total optimizer steps
batch size
learning-rate schedule
student initialization
selected hammer
corridor/exemplar allocation policy
final-test artifact
seed set
```

Wall-clock must be a generous non-binding safety ceiling.

Required matrix example:

```text
byte budgets:
  small
  medium

fixed total steps:
  25

fixed wall-clock safety ceiling:
  1800 seconds
```

Required primary comparison:

```text
two-cycle vs exemplar-only
```

under the same total artifact-byte ceiling and total optimizer-step budget.

---

# 9. Step-Controlled Family

Vary:

```text
total optimizer-step budget
```

Hold fixed:

```text
artifact subset and subset hash
artifact byte availability
student initialization
selected hammer
batch size
final-test artifact
seed set
```

Wall-clock must be non-binding.

Required invariant:

```text
same subset manifest hash across step points for a given arm and seed
```

The step-controlled family must not silently increase available artifact bytes as steps increase.

---

# 10. Time-Controlled Family

Vary:

```text
wall-clock training deadline
```

Hold fixed:

```text
artifact availability
maximum step ceiling
student initialization
selected hammer
batch size
final-test artifact
seed set
```

The runner must stop training at a safe optimizer-step boundary once the deadline is reached.

A post-hoc `runtime > ceiling` check is not sufficient.

Required fields:

```text
declared_training_deadline_seconds
deadline_start_timestamp
deadline_stop_timestamp
completed_steps_before_deadline
deadline_triggered
deadline_overshoot_seconds
```

Permit only a small documented overshoot caused by finishing the current step.

Recommended bound:

```text
deadline_overshoot_seconds <= max_observed_single_step_seconds + tolerance
```

Evaluation and checkpoint serialization time must be reported separately from controlled training time.

If exact time-controlled training is too invasive for P156.1, the time family may remain non-publication-grade, but it must not be presented as an equal-wall-clock result.

---

# 11. CLI Changes

Replace the equal-length bundled lists:

```text
--byte-budgets
--step-budgets
--wall-clock-budgets
```

with explicit family configuration.

Recommended CLI:

```bash
python scripts/run_quality_per_byte_experiment.py \
  --experiment-family bytes \
  --byte-budgets 1000000,4000000 \
  --fixed-total-steps 25 \
  --wall-clock-safety-ceiling 1800 \
  ...
```

Step family:

```bash
--experiment-family steps \
--step-budgets 10,25 \
--fixed-artifact-budget 4000000 \
--wall-clock-safety-ceiling 1800
```

Time family:

```bash
--experiment-family time \
--wall-clock-budgets 60,180 \
--fixed-artifact-budget 4000000 \
--max-step-ceiling 100000
```

A combined driver may run all three families sequentially.

---

# 12. Resume and Cache Semantics

Materialized subsets should be reusable across seeds when their selection policy is seed-independent.

Required cache key:

```text
source artifact hash
subset role
byte budget
allocation policy
selection policy
selection seed
schema version
```

Before reuse, validate:

```text
subset manifest hash
all shard hashes
record-count receipt
byte-accounting receipt
```

Do not rematerialize valid subsets unnecessarily.

Matrix-cell resume must additionally bind:

```text
experiment family
varied resource
fixed resource values
subset manifest hashes
arm config hash
seed
software commit
```

---

# 13. Required Reports

Top-level additions:

```text
budget_subset_index.json
byte_controlled_report.json
step_controlled_report.json
time_controlled_report.json
control_family_integrity.json
```

Per subset:

```text
budget_subset_manifest.json
artifact_byte_accounting.json
record_selection_receipt.json
```

Per matrix cell:

```text
control_family
varied_resource
fixed_resource_receipt
subset_hashes
actual_consumption
budget_integrity_valid
```

---

# 14. Control-Family Integrity Receipt

Required:

```json
{
  "byte_controlled": {
    "bytes_vary": true,
    "steps_fixed": true,
    "wall_clock_nonbinding": true,
    "actual_subsets_consumed": true,
    "valid": true
  },
  "step_controlled": {
    "steps_vary": true,
    "artifact_subset_fixed": true,
    "wall_clock_nonbinding": true,
    "valid": true
  },
  "time_controlled": {
    "time_varies": true,
    "artifact_subset_fixed": true,
    "step_ceiling_nonbinding": true,
    "deadline_enforced_during_training": true,
    "valid": true
  }
}
```

Any report claiming a controlled efficiency view must require the corresponding `valid` field.

---

# 15. Claims Discipline

Quality-per-byte claims require:

```text
byte-controlled family valid
actual subset consumption proven
same total step budget
same final-test artifact
aligned seeds complete
paired statistics complete
```

Quality-per-step claims require:

```text
step-controlled family valid
same subset hashes
same byte availability
```

Quality-per-time claims require:

```text
time-controlled family valid
deadline enforced during training
same artifact availability
```

If a family fails:

```text
claim_allowed = false
winner_declared = false
```

Do not derive three efficiency claims from one confounded matrix.

---

# 16. Required Tests

## Artifact Materialization

1. A byte-limited corridor subset is a valid loadable artifact.
2. A byte-limited exemplar subset is a valid loadable artifact.
3. Selected charged bytes never exceed the ceiling.
4. Selection is deterministic.
5. Different byte ceilings produce different subset hashes.
6. Shared metadata is reported separately.
7. Full training artifact is not passed into a byte-controlled runner.
8. Loader-observed subset hash matches configured subset hash.
9. Duplicate physical reads do not double-charge physical bytes.
10. Logical bytes consumed never exceed logical bytes available.

## Two-Cycle Allocation

11. Corridor and exemplar allocations sum to total ceiling.
12. Corridor overflow fails.
13. Exemplar overflow fails.
14. Combined overflow fails.
15. Unused remainder is reported.
16. Allocation policy hash is deterministic.

## Byte-Controlled Family

17. Byte budgets vary.
18. Total steps remain identical.
19. Wall-clock ceiling remains identical and non-binding.
20. Same final-test artifact is used.
21. Byte-controlled integrity passes only when actual subsets are consumed.

## Step-Controlled Family

22. Step budgets vary.
23. Artifact subset hashes remain identical.
24. Byte availability remains identical.
25. Wall-clock remains non-binding.

## Time-Controlled Family

26. Training stops at a step boundary after deadline.
27. Post-hoc-only enforcement fails integrity.
28. Overshoot is bounded and reported.
29. Evaluation time is excluded from controlled training deadline.
30. Artifact subset hashes remain identical.

## Resume

31. Valid subset cache is reused.
32. Corrupted subset shard invalidates cache.
33. Changed allocation policy invalidates resume.
34. Changed control family invalidates resume.
35. Changed fixed resource invalidates resume.

## Claims

36. Confounded bundled matrix cannot make byte/step/time claims.
37. Byte claim requires byte-family integrity.
38. Step claim requires step-family integrity.
39. Time claim requires time-family integrity.
40. Partial matrices remain non-publication-grade.

---

# 17. Required Local CPU Runs

## Fast CI Smoke

Run:

```text
1 byte-controlled point
1 seed
2–3 steps
tiny fixtures
CPU only
```

Purpose:

```text
subset materialization
real subset consumption
receipt plumbing
```

## Required CPU Scientific Matrix

Primary required matrix:

```text
experiment family: bytes
2 real byte ceilings
3 aligned seeds
4 required arms
fixed total step budget
independent final test
CPU only
```

This is the first required unconfounded P156 result.

Secondary required matrix:

```text
experiment family: steps
2 step budgets
3 aligned seeds
fixed artifact subset
CPU only
```

Time-controlled matrix may be:

```text
required if true deadline enforcement is implemented
otherwise explicitly deferred and claim-disabled
```

Codex should allow the CPU matrices to run to completion even if they take substantial local time.

---

# 18. Retrospective Labeling of Existing P156 Runs

Any result from the original bundled P156 matrix must be labeled:

```text
orchestration smoke
nominal-budget validation
confounded resource matrix
```

Required claims:

```json
{
  "actual_byte_budget_enforced": false,
  "independent_budget_views": false,
  "quality_per_byte_claim_allowed": false,
  "quality_per_step_claim_allowed": false,
  "quality_per_time_claim_allowed": false
}
```

The outputs may still be useful for:

```text
runtime estimation
pipeline debugging
resume testing
metric-path validation
```

They are not controlled efficiency evidence.

---

# 19. Non-Goals

P156.1 must not:

- modify the selected hammer;
- tune on final-test records;
- change corridor or exemplar objectives;
- add new model architectures;
- run GPT-2 scale;
- use TPU;
- use GPU;
- redesign P155;
- claim general language quality;
- claim production-scale efficiency.

---

# 20. Acceptance Criteria

P156.1 is complete when:

1. Byte ceilings materialize real standalone subsets.
2. Existing runners consume those subsets.
3. Actual charged bytes are tied to selected records.
4. Two-cycle corridor plus exemplar bytes stay under one shared ceiling.
5. Byte, step, and time controls are represented as separate experiment families.
6. The byte-controlled family varies bytes while fixing steps.
7. The step-controlled family varies steps while fixing artifact subsets.
8. Any time-controlled claim uses an enforced training deadline.
9. The original bundled matrix is labeled confounded.
10. At least two byte points, three seeds, and four arms complete locally on CPU.
11. Paired independent-final-test comparisons are produced.
12. Quality-per-byte claims remain narrow and tiny-CPU scoped.
13. P140–P155.1 regression tests remain green.

---

# 21. Expected Result

Successful primary result:

```text
phase: P156.1
status: pass
experiment_family: byte_controlled
actual_subsets_consumed: true
bytes_vary: true
steps_fixed: true
wall_clock_nonbinding: true
required_seeds_complete: true
independent_final_test: true
quality_per_byte_claim_allowed: true
result: two_cycle_better | exemplar_only_better | inconclusive
claim_scope: tiny_cpu_controlled_experiment
```

Failure example:

```text
phase: P156.1
status: fail
error: full_artifact_used_in_byte_controlled_cell
quality_per_byte_claim_allowed: false
winner_declared: false
```

After P156.1, the project may proceed to P157 only after the real byte-controlled CPU matrix is complete and honestly reported.
