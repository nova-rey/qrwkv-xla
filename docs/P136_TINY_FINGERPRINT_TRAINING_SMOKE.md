# P136 Tiny Fingerprint Training Smoke

P136 wires the P132-P135 behavioral fingerprint stack into a tiny CPU-only
training smoke. It proves that a minimal trainable student can consume a
fingerprint artifact, produce logits, compute student distribution statistics,
apply corridor loss, perform optimizer steps, emit metrics, and write simple
run artifacts.

This is an integration smoke, not a quality experiment.

## Smoke Path

```text
FingerprintTargetDataset
  -> FingerprintBatch
  -> tiny trainable position-logit student
  -> select_position_logits
  -> compute_fingerprint_distribution_stats
  -> compute_fingerprint_corridor_loss
  -> SGD update
  -> metrics/checkpoint/report JSON
```

The tiny student is a deterministic position-logit head sized from the
fingerprint artifact's `max_seq_len` and `vocab_size`. It is smoke support only
and does not represent a real QRWKV architecture.

## CLI

```bash
python scripts/run_fingerprint_smoke.py \
  --artifact tests/fixtures/behavioral_fingerprint/v0_1_valid_tiny \
  --steps 3 \
  --batch-size 2 \
  --seed 0 \
  --output-dir /tmp/qrwkv_fingerprint_p136_smoke
```

Outputs:

- `metrics.json`
- `checkpoint.json`
- `fingerprint_smoke_report.json`

## Metrics

The smoke emits the P135 metric surface:

- `fingerprint/loss_total`
- `fingerprint/loss_entropy`
- `fingerprint/loss_top1_margin`
- `fingerprint/loss_top8_mass`
- `fingerprint/loss_top32_mass`
- `fingerprint/loss_tail_mass`
- `fingerprint/inside_entropy_rate`
- `fingerprint/inside_top1_margin_rate`
- `fingerprint/inside_top8_mass_rate`
- `fingerprint/inside_top32_mass_rate`
- `fingerprint/inside_tail_mass_rate`
- `fingerprint/inside_all_rate`

## Claims

P136 proves only that the tiny fingerprint-only training stack can move on CPU
without a live teacher. It does not prove model quality, production readiness,
real teacher fingerprint generation, exemplar reservoir training, mixed
CSL/fingerprint training, Qwen support, tokenizer remapping, Pallas readiness,
or WKV/runtime math changes.
