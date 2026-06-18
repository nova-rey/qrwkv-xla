# P146 Real Student Fingerprint Training Rehearsal

P146 links the P145 tiny real-teacher fingerprint artifact producer to the
P141 `fingerprint_corridor` main runner consumer.

The rehearsal supports two modes:

- reuse an existing P145 artifact with `--fingerprint-artifact`
- build a local-files-only tiny real-teacher artifact and immediately train

The training path uses the registered `current_qrwkv` student backend through
`run_distill_stage`, performs optimizer updates, verifies that parameters move,
records finite corridor/rehearsal metrics, and writes the normal checkpoint
artifacts. The P146 report links capture metadata with training metadata and
keeps the core contract explicit:

- `teacher_real: true`
- `teacher_required_during_training: false`

The teacher is used to build the artifact. It is not used during student
training.

## CLI

Build then train from a locally cached tiny HF teacher:

```bash
python scripts/run_real_teacher_fingerprint_training_rehearsal.py \
  --build-real-teacher-artifact \
  --teacher-model sshleifer/tiny-gpt2 \
  --texts tests/fixtures/fingerprint_capture_real_teacher/tiny_texts.jsonl \
  --output-dir /tmp/qrwkv_p146_rehearsal \
  --training-steps 3 \
  --batch-size 2 \
  --learning-rate 0.01 \
  --local-files-only \
  --overwrite
```

Reuse an existing artifact:

```bash
python scripts/run_real_teacher_fingerprint_training_rehearsal.py \
  --fingerprint-artifact /tmp/qrwkv_p145_tiny_real_teacher_fingerprint \
  --output-dir /tmp/qrwkv_p146_train_existing_artifact \
  --training-steps 3 \
  --batch-size 2 \
  --overwrite
```

Reports are written as:

- `p146_rehearsal_report.json`
- `p146_rehearsal_summary.md`

## Local Cache Gate

The optional real HF integration path is local-cache only. If
`sshleifer/tiny-gpt2` is not available locally, the test skips with a clear
reason and does not download.

## Limits

P146 is an integration rehearsal. It does not claim quality improvement,
baseline parity, quality-per-byte efficiency, real-scale capture readiness, or
accelerator burn readiness.
