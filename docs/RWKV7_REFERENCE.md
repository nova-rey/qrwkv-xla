# RWKV7 Reference Student

`RWKV7ReferenceStudent` is a small JAX-only, XLA-friendly recurrent reference
implementation for exercising the QRWKV-XLA student interface and trainer path.
It is intentionally a reference core, not a final optimized RWKV7 kernel and
not an implementation of a production RWKV7 checkpoint.

## Parameterization

This chunk uses the requested matrix parameterization:

- `wr`, `wk`, `wv`, `wg`, `wo`: `[num_layers, hidden_size, hidden_size]`
- `time_decay`, `time_bias`: `[num_layers, hidden_size]`
- `embedding`: `[vocab_size, hidden_size]`

The matrix form is still lightweight for the tiny CPU-only tests because those
tests instantiate small hidden sizes and layer counts.

## Recurrence

`rwkv7_reference_layer` accepts `[batch, sequence_length, hidden_size]` inputs
and uses `jax.lax.scan` across the sequence dimension. For each token it builds
receptance, key, value, and gate projections, updates a per-channel recurrent
state with a sigmoid time decay, and returns a gated residual output.

If `attention_mask` is supplied, it is interpreted as `[batch, sequence_length]`.
Masked tokens preserve the prior recurrent state and zero the corresponding
layer output, while preserving the output shape.

## Student Interface

`RWKV7ReferenceStudent` implements `StudentModel`:

- `init_params(key)` returns a dictionary of JAX arrays.
- `apply(params, input_ids, attention_mask=None)` returns `StudentOutput`.
- `StudentOutput.hidden_states` has shape
  `[batch, num_layers, sequence_length, hidden_size]`.

The factory name for this model is `rwkv7_reference`.

## Validation Role

The reference core is used for CPU forward checks, `jax.jit` coverage,
attention-mask behavior, deterministic initialization/application tests, and
gradient coverage through the smoke training path. Optimized kernels, stateful
inference APIs, and production checkpoint compatibility are later work.
