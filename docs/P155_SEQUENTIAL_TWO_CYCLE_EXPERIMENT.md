# P155.1 Sequential Two-Cycle Split Integrity

P155 runs four controlled arms from one byte-identical shared initialization:
conventional causal-LM training, corridor-only training, exemplar-only
training, and a sequential corridor then exemplar pass. The sequential arm
crosses a finalized checkpoint boundary and starts Cycle 2 with fresh optimizer
state; corridor and exemplar objectives are never mixed.

```bash
python scripts/run_two_cycle_experiment.py \
  --training-fingerprint-artifact PATH \
  --calibration-fingerprint-artifact PATH \
  --final-test-fingerprint-artifact PATH \
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
calibration report, and publication-grade receipt. It must bind the exact
training and calibration manifests and student configuration. All three splits
must be pairwise disjoint by example ID, exact token sequence, and source text.

P153/P154 diagnostics use the calibration artifact. After arm configuration is
frozen, all checkpoints are evaluated against the same ordered final-test
records. The predeclared primary metric is dense teacher-student KL, lower is
better. Paired bootstrap intervals that include zero produce an inconclusive
result. P155 reports produced before this three-way split are validation-set
smokes only and are not independent final-test evidence.

The report separates matched exemplar-stage budgets from total arm budgets and
also records unavoidable orchestration overhead. P155 makes no formal
quality-per-byte, scale, downstream benchmark, or RADLADS parity claim.
