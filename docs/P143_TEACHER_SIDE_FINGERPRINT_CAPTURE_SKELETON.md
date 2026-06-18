# P143 Teacher-Side Fingerprint Capture Skeleton

P143 starts the producer side of the behavioral fingerprint arc. It adds a
teacher-side capture skeleton that can consume teacher-like logits from a
synthetic/logit-provider input and emit a valid `behavioral_fingerprint`
artifact.

This phase does not call a real Hugging Face teacher and does not integrate
with TOME or TeacherTextbook generation.

## Arc 2

Arc 2 is now framed as Real Student Integration + Teacher-Pure Dynamic Capture:

- P140 proved real student forward consumption of fingerprint targets.
- P141 wired fingerprint corridor training into the main runner.
- P142 verified input-conditioned student logits and parameter movement.
- P143 begins teacher-pure producer-side capture.

The next producer-side steps are P144 synthetic/fixture parity and P145 tiny
real teacher capture.

## Capture Path

The P143 skeleton implements:

- `FingerprintCaptureConfig`
- `FingerprintCaptureExample`
- synthetic fixture generation for tests and CLI smoke
- distribution statistics via the existing P134 stat math
- dynamic `stat_bands_v0` mode assignment
- per-mode min/max corridor aggregate bounds
- configurable dense-probability exemplar reservoir
- P132/P137-compatible artifact writing
- required `capture_summary.json`

The CLI entry point is:

```bash
python scripts/build_fingerprint_artifact.py \
  --synthetic-fixture tiny \
  --output-dir /tmp/qrwkv_p143_fingerprint_artifact \
  --vocab-size 16 \
  --max-seq-len 8 \
  --num-examples 4 \
  --max-exemplars 6 \
  --overwrite
```

## Dynamic Modes

P143 does not hardcode 64 or 256 modes. It assigns each processed target
position to a stat-band key:

```text
(entropy_bin, top1_margin_bin, top32_mass_bin)
```

Only observed band combinations become mode IDs. `max_modes` is a guard; if
observed combinations exceed the configured limit, capture raises a clear
error instead of silently merging modes.

## Configurable Exemplar Reservoir

The exemplar reservoir keeps up to `max_exemplars`, not exactly 1,000 examples.
The default budget can be 1,000, but the capture config owns the budget:

```yaml
exemplar_reservoir:
  enabled: true
  max_exemplars: 1000
```

For P143, selection is deterministic global top-k by:

```text
interestingness = entropy + tail_mass - top1_margin
```

Rows record reason codes such as `high_entropy`, `high_tail_mass`,
`low_margin`, or `mode_representative`.

## Artifact Output

The writer emits:

```text
manifest.json
modes.json
targets/targets-00000.jsonl
exemplars/exemplars-00000.jsonl
capture_summary.json
```

The target rows are P133-compatible corridor records. The exemplar rows are
P137-compatible `dense_probs` records. The emitted artifact validates through
the existing P132 validator and can be consumed by the existing loaders and
P140 real-student forward smoke.

## Capture Summary

Every capture writes `capture_summary.json` with:

- `phase: P143`
- `capture_method: teacher_side_capture_skeleton_v0`
- `examples_processed`
- `target_positions_processed`
- `modes_discovered`
- `records_per_mode`
- `max_exemplars`
- `exemplars_retained`
- `exemplar_reason_code_distribution`
- `artifact_validated`
- `capture_config`

## Non-Claims

P143 does not prove real teacher capture, semantic mode quality, artifact
convergence, quality improvement, baseline superiority, quality-per-byte gains,
TPU/GPU behavior, Pallas readiness, or production readiness.

P144 should tighten controlled synthetic/fixture parity. P145 should run the
first tiny real teacher capture.
