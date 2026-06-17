# P135 Corridor Loss

P135 compares P134 student distribution statistics against P133 behavioral
fingerprint target bounds. It provides pure JAX-friendly loss utilities only.
It does not wire behavioral fingerprints into a trainer, optimizer,
checkpointing path, or burn harness.

## Penalty

For a statistic value `x` and inclusive corridor bounds `[lo, hi]`:

```python
below = relu(lo - x)
above = relu(x - hi)
penalty = below**2 + above**2
```

Inside the corridor, including exact boundaries, the penalty is zero. Outside
the corridor, penalty grows quadratically with distance.

## API

```python
from qrwkv_xla.training import (
    FingerprintCorridorLossConfig,
    compute_fingerprint_corridor_loss,
    inside_bounds,
    squared_hinge_bound_penalty,
)
```

`compute_fingerprint_corridor_loss(stats, batch, config)` consumes:

- `FingerprintDistributionStats` from P134.
- `FingerprintBatch` from P133.

It returns scalar total loss, scalar per-stat losses, per-stat inside rates,
`all_inside_rate`, and `mean_weight`.

## Weights

Per-stat weights are non-negative. A zero stat weight disables that statistic's
contribution and reports that per-stat loss as zero.

When `use_record_weights=True`, per-record losses are multiplied by
`batch.weight` and normalized by `sum(weight)`. When false, record weights are
ignored and the unweighted batch mean is used.

Inside-rate diagnostics are unweighted means for interpretability.

## Claims

P135 implements only:

```text
student stats + target bounds -> loss + diagnostics
```

It does not implement P136 training smoke, trainer flags, optimizer changes,
teacher generation, exemplar reservoirs, mode-classifier losses, Pallas
promotion, Qwen/tokenizer remapping, or WKV/runtime math changes.
