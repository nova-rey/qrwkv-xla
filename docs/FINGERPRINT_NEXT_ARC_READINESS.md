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
tiny real-teacher logits through the calibrated capture path. P146 links that
artifact producer to the real registered student training path through the main
`fingerprint_corridor` runner. P147 adds the first tiny baseline comparison
harness so baseline and fingerprint arms can be recorded under shared controls.
P148 computes the first tiny corridor-adherence-per-artifact-byte reference
deltas from that scoreboard. P149 closes the arc with a constrained go/no-go
report.
P151 opens the controlled-scaling arc with a matched trained causal-LM
baseline and shared initialization contract.
P152 adds a disjoint held-out artifact contract and shared read-only evaluation
harness with paired bootstrap statistics.
P152.1 binds those measurements to the exact P151 checkpoint bytes and replaces
positional source association with explicit ID joins.

The remaining work should move from Arc 2 smoke evidence to larger controlled
fingerprint experiments with trained baseline and held-out evaluation gates.

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
- P146 real student training rehearsal from P145 artifacts
- P147 init-only baseline vs fingerprint corridor comparison harness
- P148 tiny corridor-adherence-per-artifact-byte smoke
- P149 arc report / go-with-constraints recommendation
- P151 matched trained baseline vs corridor comparison
- P152 held-out split proof and paired checkpoint evaluation
- P152.1 checkpoint lineage receipt and ID-based source provenance
- no exemplar reservoir or mixed objective in the main runner yet

Needed:

- broader held-out measurement before general method claims
- held-out tiny eval artifact before stronger quality-per-byte language
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
- P146 capture/training linkage report
- P147 artifact budget metadata in comparison reports
- P148 artifact byte denominator and delta-per-MB metrics
- P149 constraints for larger controlled experiments

Needed:

- teacher-pure exemplar reservoir selection
- artifact convergence diagnostics
- reproducible capture metadata

## Gap C - Evaluation

Current:

- plumbing smoke only
- no model quality claim
- P147 comparable tiny baseline/fingerprint scoreboard with no winner declared
- P148 corridor-adherence reference deltas per artifact byte
- P149 claims and constraints block

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

P149 should not be treated as a general quality result. It recommends
`go_with_constraints` for larger controlled experiments, not production
readiness, scale readiness, RADLADS parity, or a trained-baseline win.

P151 closes the trained-baseline gap but still does not permit a winner or
quality claim. The next gate is a distinct held-out fingerprint artifact and
shared evaluation harness.

P152 permits only a narrow winner on the predeclared held-out corridor metric.
It does not establish downstream language quality, quality per byte, scale, or
RADLADS parity. P153 measures the corridor pass as a process.
