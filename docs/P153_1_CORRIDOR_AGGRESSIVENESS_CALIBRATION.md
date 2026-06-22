# P153.1 Corridor Aggressiveness Calibration

P153.1 selects the smallest explicit corridor-only force profile that reaches
stable held-out corridor entry reliably. It invokes the P153 fixed-step runner
for every profile and seed; it does not mix objectives or start the exemplar
pass.

The ordered presets are `rock_hammer`, `ball_peen`, `sledgehammer`, and
`gallagher`. Every preset resolves loss weight, learning rate, clipping,
powered-hinge settings, per-stat weights, worst-stat boost, normalization, and
safety guards. CLI overrides are recorded in `profile_overrides_applied`.

```bash
python scripts/run_corridor_aggressiveness_calibration.py \
  --fingerprint-artifact PATH \
  --held-out-fingerprint-artifact PATH \
  --source-texts PATH \
  --output-dir PATH \
  --seeds 0,1,2 \
  --steps 100 --eval-every 5 --checkpoint-every 25
```

The top-level report, fairness contract, resolved configurations, ranking,
seed metrics, profile summaries, paired deltas, and selected reusable config
are written beside the unchanged per-seed P153 output trees. Four profiles by
one seed is a local/CI smoke. At least three complete aligned seeds per profile
are required for publication-grade calibration.

Selection gates validity and reliability before efficiency, then prefers the
less aggressive profile on equivalent evidence. An inconclusive experiment
does not force a winner. Reports make no general-quality, quality-per-byte,
scale, or RADLADS parity claim.
