# P144 Teacher Capture Synthetic Fixture Parity

P144 calibrates the P143 teacher-side capture skeleton against controlled
synthetic fixtures. P143 proved the capture path could emit valid artifacts;
P144 proves the emitted stats, modes, bounds, exemplar selection, and summaries
match known inputs.

This phase still does not call real Hugging Face teachers and does not
integrate TOME or TeacherTextbook generation.

## What Is Tested

P144 uses known probability distributions converted to logits with
`log(probs)`. The parity tests verify:

- entropy, probability-space top-1 margin, top-k mass, and tail mass
- expected `stat_bands_v0` mode keys and dynamic mode count
- per-mode record counts
- min/max corridor bounds
- `min_width` widening and stat-domain clamping
- quantile corridor bounds
- top-k interestingness exemplar selection
- stratified interestingness exemplar selection
- capture summary accuracy
- P132 validator, P133 loader, P137 loader, and P139 artifact summary
- P141 `fingerprint_corridor` consumption for one tiny step

## Quantile Bounds

P144 adds optional in-memory quantile bounds:

```yaml
corridor_bounds:
  method: quantile
  lower_quantile: 0.05
  upper_quantile: 0.95
  min_width: 1.0e-6
```

`minmax` remains the default. Quantile bounds use NumPy `quantile` over the
small in-memory P144/P145 capture records. This is not a streaming quantile
sketch.

## Stratified Exemplar Selection

P144 adds optional deterministic stratified selection:

```yaml
exemplar_reservoir:
  selection_policy: stratified_interestingness_v0
  max_exemplars: 100
  per_mode_min: 1
```

The policy attempts to retain `per_mode_min` high-interest exemplars from each
observed mode when the budget allows, then fills remaining slots by global
interestingness. It never exceeds `max_exemplars`.

The P143 default remains `top_interestingness_v0`.

## CLI

`scripts/build_fingerprint_artifact.py` now exposes the implemented parity
controls:

```bash
python scripts/build_fingerprint_artifact.py \
  --synthetic-fixture tiny \
  --output-dir /tmp/qrwkv_p144_capture_parity \
  --bounds-method quantile \
  --lower-quantile 0.25 \
  --upper-quantile 0.75 \
  --exemplar-selection-policy stratified_interestingness_v0 \
  --per-mode-min 1 \
  --overwrite
```

## Non-Claims

P144 does not prove real teacher capture, TOME integration, semantic mode
quality, artifact convergence, quality improvement, storage or compute wins,
TPU/GPU behavior, or quality-per-byte gains.

P145 should run the first tiny real teacher fingerprint capture through the
now-calibrated capture path.
