# Fingerprint Next Arc Readiness

P139 closed the tiny standalone behavioral fingerprint arc. P140 opened the
next serious arc by proving a forward-only real student backend path over
validated fingerprint corridor batches. P141 then wired corridor-only
fingerprint training into the main staged runner with optimizer, checkpoint,
metrics, report, and summary plumbing. P142 adds an input-conditioned tiny
rehearsal that verifies distinct `input_ids` change logits and optimizer steps
move real student parameters. P143 starts the producer side with a
synthetic/logit-provider teacher-side capture skeleton. P144 calibrates that
skeleton against controlled known-probability fixtures. P145 sends the first
tiny real-teacher logits through the calibrated capture path.

The remaining work should move from tiny real-teacher capture to real-student
training rehearsal.

## Gap A - Real Student / Main Runner Integration

Current:

- P141 main-runner `fingerprint_corridor` mode
- real `CurrentQRWKVStudentBackend` via the student backend registry
- input-id conditioned logits from fingerprint batches
- optimizer and checkpoint plumbing active for corridor-only fingerprint loss
- P142 rehearsal diagnostics for input conditioning and parameter movement
- P143 producer-side synthetic/logit-provider artifact emission
- P144 controlled parity for stats, modes, bounds, exemplars, summary, and P141
  consumption
- P145 tiny real-teacher capture wrapper and local-files-only CLI
- no exemplar reservoir or mixed objective in the main runner yet

Needed:

- real student training from a tiny real-teacher fingerprint artifact
- later mixed/exemplar extension if justified
- report fields that distinguish rehearsal, real runs, and science results

## Gap B - Teacher-Side Fingerprint Capture

Current:

- synthetic fixtures
- hand-authored corridor bounds and dense exemplar probabilities
- P143 synthetic/logit-provider capture skeleton
- configurable `max_exemplars`
- dynamic observed `stat_bands_v0` modes
- optional quantile bounds
- optional stratified exemplar selection
- tiny HF causal LM logits through the capture path

Needed:

- real student training rehearsal from captured real-teacher artifact
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

P145 should not be treated as a quality result. It proves that a tiny real
teacher can produce a valid behavioral fingerprint artifact through the
calibrated capture path, not that the artifact improves a student or that
large-scale capture is ready.
