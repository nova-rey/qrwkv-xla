# Fingerprint Next Arc Readiness

P139 closed the tiny standalone behavioral fingerprint arc. P140 opened the
next serious arc by proving a forward-only real student backend path over
validated fingerprint corridor batches. P141 then wired corridor-only
fingerprint training into the main staged runner with optimizer, checkpoint,
metrics, report, and summary plumbing.

The remaining work should move from this integrated training door to
input-conditioned rehearsal behavior and teacher-pure capture.

## Gap A - Real Student / Main Runner Integration

Current:

- P141 main-runner `fingerprint_corridor` mode
- real `CurrentQRWKVStudentBackend` via the student backend registry
- input-id conditioned logits from fingerprint batches
- optimizer and checkpoint plumbing active for corridor-only fingerprint loss
- no exemplar reservoir or mixed objective in the main runner yet

Needed:

- input-conditioned tiny student rehearsal over this mode
- later mixed/exemplar extension if justified
- report fields that distinguish rehearsal, real runs, and science results

## Gap B - Teacher-Side Fingerprint Capture

Current:

- synthetic fixtures
- hand-authored corridor bounds and dense exemplar probabilities

Needed:

- teacher pass that emits behavioral modes and corridor bounds
- teacher-pure exemplar reservoir selection
- artifact convergence diagnostics
- reproducible capture metadata

## Gap C - Evaluation

Current:

- plumbing smoke only
- no model quality claim

Needed:

- compare no distillation, corridor-only, exemplar-only, and mixed paths
- same-size CSL baseline
- quality-per-byte metric
- teacher KL/perplexity diagnostics

## Gap D - Artifact Scaling

Current:

- tiny JSONL fixtures
- dense exemplar probabilities

Needed:

- larger sharded artifact behavior
- streaming or low-memory loading
- compressed exemplar payload design if dense payloads are too large
- operational checks for artifact size and load time

## Boundary

P141 should not be treated as a quality result. It proves the main runner can
optimize corridor loss with a real student backend, not that the objective is
useful, teacher capture exists, or the research hypothesis is proven.
