# P139 Fingerprint Diagnostics, Reports, and Readiness

P139 closes the first behavioral fingerprint implementation arc by making the
tiny system easier to inspect. It does not add new training behavior.

## Completed Mini-Arc

- P132: `behavioral_fingerprint` artifact schema and validator.
- P133: corridor target loader and fixed-shape batches.
- P134: student distribution statistics from logits.
- P135: corridor loss against bounded statistics.
- P136/P136.1: standalone corridor-only smoke with honest pass/fail metadata.
- P137: optional dense exemplar reservoir loader and KL loss.
- P138: standalone mixed corridor plus exemplar smoke.
- P139: artifact summaries, structured smoke reports, Markdown summaries,
  canonical metric aliases, and next-arc readiness notes.

## Report Shape

Corridor-only and mixed smoke reports now include `report_schema_phase: P139`
and a report type:

- `corridor_only_smoke_report`
- `mixed_fingerprint_smoke_report`

Reports include the loaded artifact summary, training path, tiny smoke student
metadata, requested/completed optimizer steps, seed, learning rate, corridor
target summary, loss diagnostics, canonical metric sections, limitation flags,
and no-teacher/no-accelerator claims.

Mixed reports additionally include exemplar reservoir summary, loss weights,
mixed loss diagnostics, and exemplar metrics.

## Summaries

Both smoke modes write a compact Markdown summary:

```text
fingerprint_run_summary.md
```

The summary is generated from the JSON report and highlights:

- status and training path
- artifact shape
- optimizer steps and batch consumption
- loss deltas and diagnostic-only non-increase
- corridor inside rates
- exemplar KL/cross-entropy/teacher entropy when present
- limitations

## Artifact Summary

`summarize_fingerprint_artifact(...)` and
`scripts/inspect_fingerprint_artifact.py` provide a small validated summary of
fingerprint artifacts:

- artifact type/version
- teacher and tokenizer names
- vocab size and max sequence length
- tracked stats and modes
- corridor record count
- exemplar presence, payload type, and record count

## Canonical Metrics

P139 adds canonical slash-namespaced aliases while preserving existing internal
keys:

- `fingerprint/corridor/loss_total`
- `fingerprint/corridor/inside_all_rate`
- `fingerprint/exemplar/kl_loss`
- `fingerprint/exemplar/cross_entropy`
- `fingerprint/mixed/loss_total`
- `fingerprint/mixed/corridor_loss_weight`
- `fingerprint/mixed/exemplar_loss_weight`

## Proven

The P132-P139 arc proves that a CPU-safe standalone miniature can load tiny
synthetic behavioral fingerprint artifacts, train a tiny position-logit smoke
model against corridor and dense exemplar losses, emit finite metrics, and
write inspectable reports.

## Not Proven

P139 does not prove real QRWKV backend integration, main distillation runner
integration, input-conditioned student learning, teacher-side fingerprint
generation, quality improvement, artifact convergence, storage/compute wins,
TPU/GPU behavior, Pallas readiness, or production training readiness.

## Next

The recommended next arc is real student integration and teacher-pure capture.
See `docs/FINGERPRINT_NEXT_ARC_READINESS.md`.
