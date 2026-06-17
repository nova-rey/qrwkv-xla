# P134 Student Distribution Stats

P134 defines the student-side distribution statistics used by later behavioral
fingerprint corridor loss work. It is pure math over logits. It does not load
artifacts, compare values to bounds, compute loss, or modify training.

## API

```python
from qrwkv_xla.training import (
    compute_fingerprint_distribution_stats,
    compute_fingerprint_distribution_stats_at_positions,
    select_position_logits,
)
```

`compute_fingerprint_distribution_stats(logits)` expects logits shaped
`[batch, vocab]` and returns `FingerprintDistributionStats`, where each field is
shaped `[batch]`.

`select_position_logits(logits, positions)` expects logits shaped
`[batch, seq, vocab]` and positions shaped `[batch]`; it returns
`[batch, vocab]` logits using JAX-friendly array indexing.

## Definitions

All statistics are computed from `jax.nn.softmax` and `jax.nn.log_softmax`.

- `entropy`: `-sum(p * ln(p))`, using natural-log units.
- `top1_margin`: probability-space `p_top1 - p_top2`, not logit-space margin.
- `top8_mass`: cumulative probability mass of the largest 8 tokens, clamped to
  vocab size.
- `top32_mass`: cumulative probability mass of the largest 32 tokens, clamped
  to vocab size.
- `tail_mass`: `1.0 - top32_mass`, so it is zero when vocab size is at most 32
  except for tiny floating-point error.

`topk_values` is a tuple of Python integers and defaults to `(8, 32)` to keep
the shape behavior static and future XLA-friendly.

## Claims

P134 only computes student distribution statistics and selected-position logits.
It does not implement corridor loss, inside/outside-bound checks, trainer
integration, optimizer changes, exemplar reservoirs, teacher generation,
student-aware repair, Pallas changes, or WKV/runtime math changes.
