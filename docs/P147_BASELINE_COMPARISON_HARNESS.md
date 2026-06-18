# P147 Baseline Comparison Harness

P147 adds a tiny comparison harness for scoreboard infrastructure. It runs at
least two arms under shared controls:

- `baseline_init_only`: initializes the same registered student backend and
  writes a zero-step checkpoint
- `fingerprint_corridor`: trains through the existing P141/P146
  `fingerprint_corridor` path

The harness can reuse an existing P145/P146 behavioral fingerprint artifact or
build a tiny local-files-only real-teacher artifact before comparing.

## CLI

Reuse an existing artifact:

```bash
python scripts/run_fingerprint_baseline_comparison.py \
  --fingerprint-artifact /tmp/qrwkv_p145_artifact \
  --output-dir /tmp/qrwkv_p147_comparison \
  --steps 3 \
  --batch-size 2 \
  --learning-rate 0.01 \
  --seed 0 \
  --overwrite
```

Build then compare with a locally cached tiny HF teacher:

```bash
python scripts/run_fingerprint_baseline_comparison.py \
  --build-real-teacher-artifact \
  --teacher-model sshleifer/tiny-gpt2 \
  --texts tests/fixtures/fingerprint_capture_real_teacher/tiny_texts.jsonl \
  --output-dir /tmp/qrwkv_p147_comparison \
  --steps 3 \
  --batch-size 2 \
  --local-files-only \
  --overwrite
```

Reports are written as:

- `p147_comparison_report.json`
- `p147_comparison_summary.md`

## Report Contract

The JSON report records:

- artifact source, size, modes, target records, and exemplar records
- comparison controls for student backend, student config, seed, batch size,
  and eval-text fixture
- one arm record per baseline/fingerprint run
- a claims block with all quality and winner claims set to false

## Boundary

P147 does not declare a winner. It does not claim model quality improvement,
quality-per-byte efficiency, RADLADS parity, or benchmark validity. P148 is the
first phase intended to ask quality-per-byte questions.
