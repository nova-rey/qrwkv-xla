# P152 Held-Out Fingerprint Evaluation

P152 evaluates the two matched-budget P151 checkpoints against the same
disjoint behavioral fingerprint records without performing optimizer updates.

Capture training and held-out artifacts with different example-ID prefixes,
then write deterministic provenance sidecars:

```bash
python scripts/write_fingerprint_provenance.py \
  --artifact artifacts/train_fingerprint \
  --source-file data/train.jsonl \
  --artifact-role training

python scripts/write_fingerprint_provenance.py \
  --artifact artifacts/held_out_fingerprint \
  --source-file data/held_out.jsonl \
  --artifact-role held_out_evaluation
```

Run evaluation:

```bash
python scripts/run_fingerprint_held_out_evaluation.py \
  --baseline-checkpoint artifacts/p151/baseline/checkpoints/final \
  --fingerprint-checkpoint artifacts/p151/fingerprint/checkpoints/final \
  --held-out-fingerprint-artifact artifacts/held_out_fingerprint \
  --train-fingerprint-artifact artifacts/train_fingerprint \
  --p151-report artifacts/p151/trained_baseline_comparison_report.json \
  --output-dir artifacts/p152_held_out_eval \
  --bootstrap-samples 1000 \
  --overwrite
```

Evaluation fails closed for missing or invalid provenance, overlapping IDs,
source texts or token sequences, incompatible artifact contracts, invalid P151
fairness, incompatible checkpoints, non-finite metrics, or record-order drift.

The predeclared primary metric is mean held-out corridor loss, where lower is
better. A winner is declared only when the paired bootstrap interval excludes
zero; otherwise the result is `inconclusive`. The claim applies only to the
declared held-out fingerprint metric and does not establish language quality,
quality per byte, scale readiness, or RADLADS parity.
