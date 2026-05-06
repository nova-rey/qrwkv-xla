# RWKV7 Qwen Reference Parameter Surface

Phase P33 records the tiny deterministic parameter surface for
`rwkv7_qwen_reference`. This is fixture and regression coverage for the local
slow JAX reference backend. It is not a RADLADS checkpoint compatibility claim
and it is not full RADLADS numerical parity.

## Tiny Fixture Config

The checked-in fixture bundle lives at `tests/fixtures/qwen_reference/` and can
be regenerated offline with:

```bash
python scripts/generate_qwen_reference_fixtures.py --out tests/fixtures/qwen_reference --seed 1234 --overwrite
```

The P33 handoff smoke may also generate under `artifacts/`:

```bash
python scripts/generate_qwen_reference_fixtures.py --out artifacts/p33_qwen_reference_fixtures --seed 1234 --overwrite
```

Fixture config:

- backend: `rwkv7_qwen_reference`
- seed: `1234`
- dtype: `float32`
- vocab size: `32`
- hidden size: `8`
- layers: `2`
- attention heads: `2`
- KV heads: `1`
- batch size: `2`
- sequence length: `5`
- logits: enabled
- mixer outputs: enabled

## Payloads

Each payload stores full-sequence and stepwise arrays:

- `input_ids`
- optional `attention_mask`
- `full_hidden_states`
- `full_logits`
- `full_mixer_outputs`
- `full_wkv_matrix_state`
- `full_shift_state`
- `full_next_position`
- `step_hidden_states`
- `step_logits`
- `step_mixer_outputs`
- `step_wkv_matrix_state`
- `step_shift_state`
- `step_next_position`

The manifest records deterministic payload hashes and max absolute/relative
full-vs-step differences. Current fixture tolerances are `atol=1e-5` and
`rtol=1e-5`.

## Mask Cases

P33 records three mask edge fixtures:

- `no_mask`: no `attention_mask` argument; every token is active.
- `interior_mask`: masked interior tokens in both batch rows.
- `prefix_left_padding`: prefix/left-padding shape case using a `[B,T]` mask.

Current masked-token behavior is documented rather than changed:
`attention_mask` is shaped `[B,T]`. Masked tokens zero the value stream,
attention output, and MLP output for that token. Recurrence decay/update from
the previous matrix state still runs, `shift_state` is updated to the masked
token representation, and `next_position` advances by sequence length.

The prefix/left-padding fixture is a shape and current-behavior fixture only.
It does not claim separate left-padding position semantics; masked prefix
positions still consume positions.

## Parameter Surface

The tiny deterministic snapshot currently contains 19 parameter leaves:

| Path | Shape |
| --- | --- |
| `final_layernorm.weight` | `[8]` |
| `layers.input_layernorm.weight` | `[2, 8]` |
| `layers.mlp.down_proj.weight` | `[2, 32, 8]` |
| `layers.mlp.gate_proj.weight` | `[2, 8, 32]` |
| `layers.mlp.up_proj.weight` | `[2, 8, 32]` |
| `layers.post_attention_layernorm.weight` | `[2, 8]` |
| `layers.self_attn.a_proj.weight` | `[2, 8, 8]` |
| `layers.self_attn.b_proj.weight` | `[2, 8, 8]` |
| `layers.self_attn.g_proj.weight` | `[2, 8, 8]` |
| `layers.self_attn.k_proj.weight` | `[2, 8, 4]` |
| `layers.self_attn.o_proj.weight` | `[2, 8, 8]` |
| `layers.self_attn.q_proj.weight` | `[2, 8, 8]` |
| `layers.self_attn.time_bias` | `[2, 8]` |
| `layers.self_attn.time_mix` | `[2, 8]` |
| `layers.self_attn.v_proj.weight` | `[2, 8, 4]` |
| `layers.self_attn.w_proj.weight` | `[2, 8, 8]` |
| `lm_head.bias` | `[32]` |
| `lm_head.weight` | `[8, 32]` |
| `token_embedding.weight` | `[32, 8]` |

The fixture tests require stable flattened paths, grouped shapes, finite hashes,
and no anonymous toy-shell naming.

## Boundaries

P33 does not add Pallas kernels, TPU compilation/profiling, `pjit`, model
sharding, Qwen-scale runs, HF export, `lm_eval`, WandB, RADLADS checkpoint
compatibility, or any claim of full RADLADS numerical parity.
