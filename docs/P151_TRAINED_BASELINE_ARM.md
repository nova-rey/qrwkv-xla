# P151 Trained Baseline Arm

P151 compares a conventional trained causal-LM student with the existing
`fingerprint_corridor` training path under a matched training budget.

```bash
python scripts/run_fingerprint_trained_baseline_comparison.py \
  --fingerprint-artifact artifacts/fingerprint \
  --source-texts tests/fixtures/fingerprint_capture_real_teacher/tiny_texts.jsonl \
  --output-dir artifacts/p151_trained_baseline \
  --steps 3 \
  --batch-size 2 \
  --optimizer adamw \
  --learning-rate 0.0001 \
  --overwrite
```

The artifact and source file must describe the same deterministic example-ID
sequence. Both arms resume from one shared step-zero checkpoint, and the report
is invalid unless their initial parameter fingerprints and all required budget
predicates match.

Source JSONL rows must normally contain both `example_id` and `text`. Legacy
text-only fixtures require `--allow-legacy-positional-source-join`; that mode
records reduced lineage confidence and is not publication-grade.

Outputs include the combined report, metrics, and summary plus canonical
`baseline/` and `fingerprint/` checkpoint/report trees. This phase covers only
Cycle 1. It does not enable exemplar training, held-out evaluation, winner
selection, or general quality claims.
