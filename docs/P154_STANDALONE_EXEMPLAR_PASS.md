# P154 Standalone Exemplar Pass

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
artifact, sampling policy, and optimizer family.

These artifacts establish standalone exemplar-pass execution only. They do not
claim full two-cycle superiority, general model quality, scaling behavior,
quality-per-byte superiority, or RADLADS parity.
