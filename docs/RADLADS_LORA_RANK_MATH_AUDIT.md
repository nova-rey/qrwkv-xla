# RADLADS LoRA Rank Math Audit

Source audited:
`/home/nyx/.openclaw/workspace/_refs/RADLADS/rwkv7qwen2/modeling_rwkv7qwen2.py`

Observed active formulas used for P48:

| Source behavior | P48 status |
| --- | --- |
| `w_lora_result = w0 + (tanh(xw @ w1) @ w2).float()` | Implemented behind `radlads_low_rank_decay` / `radlads_compatible_math` |
| `a = sigmoid(a0 + (xa @ a1) @ a2)` | Implemented behind `radlads_low_rank_iclr` / `radlads_compatible_math` |
| layer 0 sets `v_first = v`; later layers apply value residual mix | Implemented behind `radlads_value_residual_mix` / `radlads_compatible_math` |
| `balance_state` branch normalizes `k`, otherwise uses `k_k` and `k_a` | Implemented behind `radlads_balance_state_terms` / `radlads_compatible_math` |
| attention output uses `group_norm(... ln_x.weight, ln_x.bias, eps=head_dim * 1e-5)` when enabled, otherwise source scales by `head_dim ** -0.5` | Group norm implemented behind `radlads_attention_group_norm`; source scale is applied in overall compatible mode when group norm is off |
| `r_k` parameter exists | Represented as a parameter leaf only |

Important caveat: the inspected source contains an `r_k` residual expression,
but that line is commented out. P48 therefore does not activate `r_k` math and
does not report it as active parity.

Legacy/default behavior remains unchanged by default. The new parameter leaves
are initialized consistently in `init_params`, but the slow attention path only
uses them when the explicit RADLADS flags select the source-backed branches.

Smoke/report command:

```bash
python scripts/run_radlads_lora_rank_math_smoke.py
```

Expected artifact directory:
`artifacts/p48_radlads_lora_rank_math`

Expected files:

- `P48_RESULTS.md`
- `lora_rank_math_report.json`
- `P48_PARAMETER_SURFACE_MAP.md`
- `parameter_surface_map.json`
