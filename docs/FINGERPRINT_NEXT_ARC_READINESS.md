# Fingerprint Next Arc Readiness

P139 closed the tiny standalone behavioral fingerprint arc. P140 opens the next
serious arc by proving a forward-only real student backend path over validated
fingerprint corridor batches. The remaining work should move from that
standalone forward smoke to main-runner integration and teacher-pure capture.

## Gap A - Real Student / Main Runner Integration

Current:

- P140 standalone real `CurrentQRWKVStudentBackend` forward smoke
- input-id conditioned logits from the registered student backend
- no main staged distillation runner integration
- no optimizer or checkpoint training semantics in the fingerprint path

Needed:

- main `run_distill_stage` integration or equivalent
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

P140 should not be treated as a quality result. It proves real-student forward
compatibility for fingerprint diagnostics, not training, teacher capture,
main-runner integration, or the research hypothesis.
