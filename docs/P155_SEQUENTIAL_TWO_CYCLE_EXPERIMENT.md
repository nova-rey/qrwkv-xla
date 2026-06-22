# P155 Sequential Two-Cycle Experiment

P155 runs four controlled arms from one byte-identical shared initialization:
conventional causal-LM training, corridor-only training, exemplar-only
training, and a sequential corridor then exemplar pass. The sequential arm
crosses a finalized checkpoint boundary and starts Cycle 2 with fresh optimizer
state; corridor and exemplar objectives are never mixed.

```bash
python scripts/run_two_cycle_experiment.py \
  --training-fingerprint-artifact PATH \
  --held-out-fingerprint-artifact PATH \
  --source-texts PATH \
  --selected-profile-receipt PATH \
  --output-dir PATH \
  --student-backend current_qrwkv \
  --baseline-steps 100 \
  --corridor-steps 100 \
  --exemplar-steps 100 \
  --batch-size 1 \
  --bootstrap-samples 1000 \
  --bootstrap-seed 0
```

The selected-profile receipt must sit beside its resolved profile config,
calibration report, and publication-grade receipt. All checkpoints are
evaluated against the same ordered held-out records. The predeclared primary
metric is dense teacher-student KL, lower is better. Paired bootstrap intervals
that include zero produce an inconclusive result.

The report separates matched exemplar-stage budgets from total arm budgets and
also records unavoidable orchestration overhead. P155 makes no formal
quality-per-byte, scale, downstream benchmark, or RADLADS parity claim.
