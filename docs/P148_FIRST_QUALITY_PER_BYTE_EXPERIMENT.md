# P148 First Quality-Per-Byte Experiment

P148 runs the first tiny corridor-adherence-per-artifact-byte smoke experiment.
It reuses the P147 comparison harness, evaluates the init-only reference and
fingerprint corridor checkpoints against the fingerprint corridor artifact, and
computes simple delta-per-byte diagnostics.

The current experiment uses:

- `baseline_init_only` as a reference baseline
- `fingerprint_corridor` as the trained fingerprint method
- `train_artifact_reuse` as the eval split
- `corridor_adherence` as the quality proxy

Because the baseline is init-only, this is a reference delta, not a fair
method-vs-method comparison.

## CLI

Reuse an existing artifact:

```bash
python scripts/run_fingerprint_quality_per_byte_experiment.py \
  --fingerprint-artifact /tmp/qrwkv_p145_artifact \
  --output-dir /tmp/qrwkv_p148_quality_per_byte \
  --steps 3 \
  --batch-size 2 \
  --learning-rate 0.01 \
  --seed 0 \
  --eval-split train_artifact_reuse \
  --overwrite
```

Build then run with a locally cached tiny HF teacher:

```bash
python scripts/run_fingerprint_quality_per_byte_experiment.py \
  --build-real-teacher-artifact \
  --teacher-model sshleifer/tiny-gpt2 \
  --texts tests/fixtures/fingerprint_capture_real_teacher/tiny_texts.jsonl \
  --output-dir /tmp/qrwkv_p148_quality_per_byte \
  --steps 3 \
  --batch-size 2 \
  --local-files-only \
  --overwrite
```

Reports are written as:

- `p148_quality_per_byte_report.json`
- `p148_quality_per_byte_summary.md`

## Metrics

The report records per-arm corridor adherence:

- `eval/corridor_loss_total`
- `eval/corridor_inside_all_rate`
- `eval/corridor_inside_entropy_rate`
- `eval/corridor_inside_top1_margin_rate`
- `eval/corridor_inside_top8_mass_rate`
- `eval/corridor_inside_top32_mass_rate`
- `eval/corridor_inside_tail_mass_rate`

It also computes:

- absolute and relative corridor loss delta versus init-only reference
- corridor loss delta per fingerprint artifact MB
- inside-all rate delta per fingerprint artifact MB

## Boundary

P148 does not declare a winner. It does not claim general quality improvement,
RADLADS parity, scale readiness, or statistically meaningful quality-per-byte
advantage. P149 should use this tiny evidence as one input to the arc report.
