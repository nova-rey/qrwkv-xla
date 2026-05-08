# RADLADS LoRA Rank Math

P48 adds source-shaped RADLADS low-rank math surfaces to the
`rwkv7_qwen_reference` slow JAX backend. The default configuration remains the
legacy QRWKV-XLA slow reference path. RADLADS-compatible math is only selected
through explicit flags, or through `radlads_compatible_math=True`.

Implemented flagged surfaces:

| Surface | Config gate | Slow-reference behavior |
| --- | --- | --- |
| `w0/w1/w2` | `radlads_low_rank_decay` or `radlads_compatible_math` | `w0 + (tanh(xw @ w1) @ w2).float()` |
| `a0/a1/a2` | `radlads_low_rank_iclr` or `radlads_compatible_math` | `sigmoid(a0 + (xa @ a1) @ a2)` |
| `v0/v1/v2` | `radlads_value_residual_mix` or `radlads_compatible_math` | layer 0 exports `v_first`; later layers mix `v + (v_first - v) * sigmoid(v0 + (xv @ v1) @ v2)` |
| `k_k/k_a` | `radlads_balance_state_terms` or `radlads_compatible_math` | source-backed balance-state branch for normalized `kk` and adjusted `k` |
| `ln_x.weight/bias` | `radlads_attention_group_norm` | group norm over attention heads with `eps=head_dim * 1e-5`; otherwise compatible mode applies the source `head_dim ** -0.5` scale |
| `r_k` | represented only | parameter leaf exists; math remains inactive because the inspected RADLADS residual line is commented out |

Rank defaults follow the inspected RADLADS source helper:

```text
max(1, round(hidden_size ** exponent * multiplier / 32)) * 32
```

The defaults are `exponent=0.5, multiplier=1.8` for decay and ICLR, and
`exponent=0.5, multiplier=1.3` for value residual mixing. Tests and the P48
smoke use smaller explicit ranks to keep CPU execution small.

This phase does not claim full RADLADS parity, optimized kernel parity, fitted
weight conversion, or Qwen-scale execution. It completes the slow-reference math
surface needed for later audited parity work.
