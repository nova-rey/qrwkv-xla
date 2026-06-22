# P153 Corridor-Pass Measurement

P153 measures Cycle 1 as a process rather than a final training smoke. It logs
training and held-out corridor progress from step 0 at a fixed cadence, detects
strict, threshold, and stable corridor entry, writes periodic checkpoints, and
accounts for record visits, tokens, logical artifact bytes, physical artifact
size, and wall-clock categories.

Training and held-out artifacts require publication-grade P152.1 provenance
sidecars with explicit source IDs and a disjoint split.

```bash
python scripts/run_corridor_measurement.py \
  --fingerprint-artifact artifacts/train_fingerprint \
  --held-out-fingerprint-artifact artifacts/held_out_fingerprint \
  --source-texts data/train.jsonl \
  --output-dir artifacts/p153_corridor_measurement \
  --steps 100 \
  --eval-every 5 \
  --checkpoint-every 25 \
  --batch-size 1 \
  --optimizer adamw \
  --learning-rate 0.0001 \
  --corridor-entry-threshold 0.95 \
  --stable-entry-evals 3 \
  --overwrite
```

Fixed-step mode is the default. `--stop-on-stable-entry` stops only after the
predeclared number of consecutive held-out evaluations meets the threshold.
Step 0, every configured interval, and the final step are always evaluated.

Required outputs include the report, summary, JSONL trajectory, efficiency,
entry, resource, and lineage receipts plus step-zero, periodic, and final
checkpoints. P153 reports corridor progress per resource unit as a process
metric; it does not claim final language quality or quality-per-byte
superiority.
