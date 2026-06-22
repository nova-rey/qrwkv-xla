# P154.1.1 Standalone Exemplar Pass Integrity

P154 implements Cycle 2 of the behavioral fingerprint training design. The
runner loads parameters from a completed Cycle 1 corridor checkpoint, rejects
lineage mismatches, initializes fresh optimizer state, and optimizes only the
dense-probability exemplar KL objective. Corridor loss, causal-LM loss, and
mixed objectives remain disabled.

```bash
python scripts/run_exemplar_pass.py \
  --corridor-checkpoint PATH \
  --fingerprint-artifact PATH \
  --held-out-fingerprint-artifact PATH \
  --output-dir PATH \
  --student-backend current_qrwkv \
  --steps 100 \
  --batch-size 1 \
  --optimizer adamw \
  --learning-rate 0.00005 \
  --max-grad-norm 1.0 \
  --exemplar-sampling-policy sequential \
  --eval-every 5 \
  --checkpoint-every 25
```

The runner writes a step-zero evaluation, cadence and final trajectory rows,
best and final exemplar checkpoints, sampling and resume receipts, resource
accounting, lineage validation, and optional corridor-retention diagnostics.
Resume checkpoints must belong to Cycle 2 and match the same corridor parent,
artifact, resolved record order, record limit, batch size, sampling seed,
optimizer family, learning rate, and gradient clipping configuration. Supplying
a held-out artifact requires a non-empty held-out exemplar reservoir; training
exemplars are never used as an implicit fallback. Optional P153 and calibration
receipts must identify the exact parent checkpoint and parameters.

Corridor retention degradation and corridor exit are reported separately. An
exit requires the initial inside rate to meet the configured threshold and a
later evaluation to fall below it.

These artifacts establish standalone exemplar-pass execution only. They do not
claim full two-cycle superiority, general model quality, scaling behavior,
quality-per-byte superiority, or RADLADS parity.
