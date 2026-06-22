# P153.1.1 Aggressiveness Selection Safety Cleanup

P153.1 establishes the four explicit corridor-force presets. P153.1.1 keeps
that fixed-step calibration matrix intact and closes the remaining selection
safety gap: fast destructive, incomplete, invalid, or too-weak profiles are
reported but cannot win.

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
seed metrics, profile summaries, paired deltas, aggregate validation receipt,
profile selection receipt, publication-grade receipt, and selected reusable
config are written beside the unchanged per-seed P153 output trees. Four
profiles by one seed is a local/CI smoke. At least three complete aligned
seeds per profile are required for publication-grade calibration.

Selection order is explicit: validity, completeness, reliability, efficiency,
then minimum sufficient force. Paired bootstrap tie handling is deterministic;
if the primary efficiency delta is inconclusive, the lower aggressiveness rank
wins. An inconclusive experiment does not force a winner. Reports make no
general-quality, quality-per-byte, scale, or RADLADS parity claim.
