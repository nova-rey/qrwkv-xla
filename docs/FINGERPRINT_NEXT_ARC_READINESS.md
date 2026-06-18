# Fingerprint Next Arc Readiness

P139 closes the tiny standalone behavioral fingerprint arc. The next serious
arc should move from synthetic standalone smoke to real student and teacher
capture paths.

## Gap A - Real Student / Main Runner Integration

Current:

- standalone tiny position-logit smoke
- no input-id conditioning
- no main staged distillation runner integration
- no real QRWKV/Radjax backend training

Needed:

- real QRWKV/Radjax student backend path
- main `run_distill_stage` integration or equivalent
- input-conditioned logits
- real optimizer/checkpoint loop
- report fields that distinguish smoke, rehearsal, and real runs

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

The next arc should not treat P139 as a quality result. P139 makes the tiny
system legible. It does not prove the research hypothesis.
