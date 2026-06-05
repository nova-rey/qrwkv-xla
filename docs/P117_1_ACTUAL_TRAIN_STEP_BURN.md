# P117.1 Actual Train-Step Burn

## Purpose

P117.1 makes real mode in the first serious burn harness execute actual
optimizer updates against a dense TeacherTextbook.

## Why P117 Was Not Enough

P117 proved allocation, launch, environment setup, TeacherTextbook validation,
readiness, and real-mode gating, but the harness still reported
`steps_completed=0`. P117.1 removes that pass path.

## Train-Step Knobs

The harness supports:

```text
--max-steps INT
--batch-size INT
--allow-textbook-reuse / --no-allow-textbook-reuse
--teacher-textbook PATH
```

## Dataset Exhaustion and Reuse

When reuse is disabled, the harness fails before training if
`max_steps * batch_size` exceeds available examples. When reuse is enabled, the
loader cycles examples and reports consumed, unique, and reused counts.

## Immediate 8-Step Configuration

```bash
python scripts/run_first_serious_burn.py \
  --output ~/qrwkv_artifacts/p117_1_real_train_steps \
  --mode real \
  --confirm-serious-burn \
  --teacher-textbook ~/qrwkv_artifacts/p117_teacher_textbook_tiny_gpt2_smoke \
  --max-steps 8 \
  --batch-size 1 \
  --no-allow-textbook-reuse
```

## Future 100-Step Configuration

```bash
python scripts/run_first_serious_burn.py \
  --output ~/qrwkv_artifacts/p117_1_100step_reuse_smoke \
  --mode real \
  --confirm-serious-burn \
  --teacher-textbook ~/qrwkv_artifacts/p117_teacher_textbook_tiny_gpt2_smoke \
  --max-steps 100 \
  --batch-size 4 \
  --allow-textbook-reuse
```

## Real-Mode Failure Invariant

Real mode may not pass unless at least one train step completes, finite loss is
recorded, parameters change, and a non-empty checkpoint is written.

## Report Fields

Reports include step/batch/reuse counts, TeacherTextbook metadata, initial and
final loss, loss trace path, checkpoint path, backend/devices, hostname, and
`real_training_executed`.

## Claims Not Made

P117.1 does not claim model quality, long-run training readiness, Qwen/Gemma
support, Rosetta/Vocab C/tokenizer remapping, full HF-native integration, TPU
multi-worker hardening, WKV math changes, or Pallas promotion.
