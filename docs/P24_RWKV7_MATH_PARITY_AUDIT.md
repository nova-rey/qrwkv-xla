# P24 RWKV7 Math Parity Audit

## Status

`src/qrwkv_xla/students/rwkv7_reference.py` is a simplified placeholder, not
parity-aligned with RADLADS RWKV7 math.

This is not a failure of the current smoke path: earlier phases documented this
student as a CPU-safe, XLA-friendly reference core for shape contracts,
masking, JIT, gradients, and distillation wiring. P24 confirms it should not be
treated as a numerically compatible implementation of the RADLADS RWKV7
attention recurrence or as a checkpoint-compatible Qwen/RWKV7 block.

## Files Audited

- `src/qrwkv_xla/students/rwkv7_reference.py`
- `tests/test_rwkv7_reference_block.py`
- `tests/test_rwkv7_reference_student.py`
- `docs/RWKV7_REFERENCE.md`
- `/home/nyx/.openclaw/workspace/_refs/RADLADS/rwkv7qwen2/modeling_rwkv7qwen2.py`
- `/home/nyx/.openclaw/workspace/_refs/RADLADS/rwkv7_attn_triton.py`
- `/home/nyx/.openclaw/workspace/_refs/RADLADS/rwkv7_attn_triton_bighead.py`
- `/home/nyx/.openclaw/workspace/_refs/RADLADS/cuda/wkv7_cuda.cu`
- `/home/nyx/.openclaw/workspace/_refs/RADLADS/cuda/wkv7g_v1.cu`
- `/home/nyx/.openclaw/workspace/_refs/RADLADS/rwkv_cuda_wind/wind_rwkv7.cu`

## Equation Alignment

RADLADS computes RWKV7 attention from projected tensors `r`, `log_w`, `k`, `v`,
`a`, and `b`, shaped by batch, time, head, and head dimension. The recurrent
core keeps a per-head matrix state `[B, H, N, N]`. The recurrent update includes
decay by `w = exp(-exp(log_w_like_input))`, a value-key outer-product term, and
an in-context learning term derived from `a` and `b`:

- CUDA scalar form: `state[j] = state[j] * w[j] + k[j] * v + sa * b[j]`,
  where `sa = dot(a, state)`.
- Model fallback/comment form: `state = state * w + state @ ab + vk`, with
  `vk = v outer k` and `ab = (-kk) outer (kk * a)`.
- Triton chunk form: uses prefix products, lower-triangular solves, and final
  state updates equivalent to the same matrix-state recurrence.

The current JAX reference uses a scalar channel state `[B, hidden_size]`:

`state = sigmoid(time_decay) * state + key * value`

and emits:

`output = tanh(token + ((sigmoid(token @ wr + bias) * state * sigmoid(token @ wg)) @ wo))`

That omits RADLADS head splitting, matrix state, `a`/`b` in-context term,
`exp(-exp(.))` decay parameterization, normalized `kk`, value residual mixing,
group norm / head scaling parity, RoPE on `r`/`k`, and Qwen decoder residual
and MLP context. It is therefore a placeholder recurrence, not an aligned
equation implementation.

## State Update Semantics

P24 adds a minimal state API to `rwkv7_reference_layer`:

- `initial_state`: optional `[batch, hidden_size]` scalar-channel state.
- `return_state`: returns the final scalar state without changing existing
  callers.

This only exposes the current placeholder state. It is not RADLADS cache parity,
because RADLADS stores per-layer `kv_state` as `[B, H, N, N]` plus a shift
state. The current student has no shift-state semantics and no production
generation cache contract.

The new tests verify current all-at-once vs token-by-token final-state
equivalence for the placeholder recurrence.

## Masking Semantics

RADLADS applies a two-dimensional padding mask by zeroing `v` on masked tokens
before recurrence. That means masked tokens do not contribute value content, but
other projected quantities still participate according to the RADLADS kernel
path and state equations.

The current JAX reference preserves previous scalar state when
`attention_mask == 0` and zeros the token output and mixer output at that
position. This is useful for local smoke tests but is not documented as
RADLADS-equivalent masking. The P24 tests pin the current semantics only.

## Batch Semantics

Both RADLADS and the current JAX reference keep batch rows independent. P24 adds
batched-vs-unbatched equivalence tests for the current placeholder recurrence.
This is a genuine invariant for the current implementation and does not imply
checkpoint or equation parity.

## JIT Semantics

The current implementation is JAX-native and uses `jax.lax.scan`, so it is
compatible with `jax.jit` for static parameter shapes. P24 adds eager-vs-jit
equivalence coverage including returned final state.

RADLADS Triton/CUDA kernels are GPU-oriented and are not imported or executed
by this phase.

## Gradient Sanity

The current JAX reference is differentiable through its placeholder recurrence.
P24 adds finite-gradient coverage and a tiny SGD optimizer step with no NaNs.
This validates local training stability for small deterministic fixtures, not
RADLADS backward-kernel parity.

## Parameter Mapping Risk

Parameter mapping risk is high.

The current parameter set is:

- `wr`, `wk`, `wv`, `wg`, `wo`
- `time_decay`, `time_bias`
- `embedding`
- optional local LM head

RADLADS uses Qwen/RWKV7 attention projections and parameters such as `q_proj`,
`k_proj`, `v_proj`, `o_proj`, `w0/w1/w2`, `a0/a1/a2`, value residual mixing
parameters, gate variants, `k_k`, `k_a`, `r_k`, optional group norm, RoPE, and
Qwen decoder components. There is no safe direct checkpoint mapping from
RADLADS into the current JAX placeholder.

## Numerical Tolerances

P24 tests use tiny deterministic `float32` fixtures and compare the current JAX
implementation against an independent NumPy mirror with `atol=1e-6`. Eager vs
JIT and chunked-vs-token stepping are expected to match at the same tolerance
for these fixtures.

No tolerance is claimed against RADLADS torch, Triton, or CUDA outputs because
the equations and state shape are not aligned.

## P24 Test Harness

The local parity harness is
`src/qrwkv_xla/students/rwkv7_reference_parity_harness.py`. It is pure NumPy
and mirrors only the current JAX placeholder recurrence. The focused tests in
`tests/test_rwkv7_math_parity.py` cover:

- NumPy harness vs JAX output and final state.
- All-at-once vs token-by-token final-state equivalence.
- Batched vs unbatched equivalence.
- Eager vs `jax.jit` equivalence.
- Finite gradients.
- Tiny optimizer step no-NaN behavior.

## Conclusion

The current `rwkv7_reference` remains suitable for CPU-only QRWKV-XLA plumbing
tests and conservative training smoke coverage. It must be labelled a
simplified placeholder until a future phase implements RADLADS-aligned
head-wise matrix-state RWKV7 recurrence, parameter mapping, and parity tests
against reference outputs.
