# RADLADS RWKV7 Gap Audit

Phase P31 audits `rwkv7_radlads_reference` against the local RADLADS
RWKV7Qwen2 reference. This is an audit and planning document, not an
implementation plan for kernels.

## Summary

QRWKV-XLA has a useful JAX reference backend for the RADLADS recurrence shape:
head-wise matrix state, scan-based token recurrence, RADLADS-style decay,
normalized key direction, in-context update terms, recurrent readout, finite
gradient coverage, logits plumbing, checkpoint resume coverage, and tiny
teacher-target distillation smokes.

It is still a RADLADS-shaped JAX reference backend, not full RADLADS parity.
The missing work is mostly model math and block compatibility, not optimized
kernel work. The current backend is not ready to be treated as the target for
Pallas, TPU kernel, `pjit`, sharding, export, or quality claims.

Recommended P32 scope: finish the slow JAX compatibility target around the
RADLADS attention/block contract before any kernel work. The minimum useful P32
shape is Qwen-style decoder block compatibility, RADLADS parameter naming and
shape mapping, cache state contract with both KV and shift state, and focused
math tests around the current recurrent equations.

Classification labels used by this audit are constrained to the following
status values: `implemented`, `partially implemented`, `missing`,
`intentionally deferred`, `not applicable`, and
`unknown / needs source confirmation`.

Priority values are constrained to: `P32 must-fix before kernel`,
`P33 near-term quality/compatibility`, `export-only concern`,
`scale-only concern`, `documentation-only`, and `not needed`.

## Current QRWKV-XLA State

The current `rwkv7_radlads_reference` backend lives in
`src/qrwkv_xla/students/rwkv7_radlads_reference.py`.

Current implemented surface:

- Student factory architecture: `rwkv7_radlads_reference`.
- State shape: per layer `[batch, heads, head_size, head_size]`, exposed across
  layers as `[layers, batch, heads, head_size, head_size]`.
- Projections: `wr`, `ww`, `wk`, `wv`, `wa`, `wb`, `wg`, `wo`, plus
  `time_bias`.
- Per-token recurrence: normalize `k` into `kk`, compute
  `log_w = -exp(-0.5) * sigmoid(w)`, decay the matrix state, apply
  `prev_state @ ab`, inject `vk`, read with `r`, gate with `sigmoid(wg)`, and
  emit `tanh(token + mixer)`.
- Padding behavior: `attention_mask` zeroes the value stream and output/mixer,
  while recurrence decay and in-context update still run.
- Distillation integration: hidden MSE, optional logits KL, optional mixer
  output plumbing, checkpoint save/resume, and tiny fake/HF-shaped smokes.

Current missing surface:

- No explicit RoPE.
- No Qwen RMSNorm/MLP decoder block skeleton.
- No grouped KV heads.
- No shift-state cache.
- No LoRA-rank factorization matching RADLADS names.
- No parameter mapping compatibility layer.
- No numerical comparison against RADLADS PyTorch, Triton, or CUDA outputs.

The existing distill pipeline proves hidden/logit loss plumbing. It does not
prove RADLADS parity.

## RADLADS Reference Summary

The local RADLADS reference inspected for this audit is:

- `/home/nyx/.openclaw/workspace/_refs/RADLADS/rwkv7qwen2/modeling_rwkv7qwen2.py`
- `/home/nyx/.openclaw/workspace/_refs/RADLADS/rwkv7_attn_triton.py`
- `/home/nyx/.openclaw/workspace/_refs/RADLADS/rwkv7_attn_triton_bighead.py`
- `/home/nyx/.openclaw/workspace/_refs/RADLADS/cuda/wkv7_cuda.cu`
- `/home/nyx/.openclaw/workspace/_refs/RADLADS/cuda/wkv7g_v1.cu`
- `/home/nyx/.openclaw/workspace/_refs/RADLADS/rwkv_cuda_wind/wind_rwkv7.cu`
- `/home/nyx/.openclaw/workspace/_refs/RADLADS/rwkv7qwen2/configuration_rwkv7qwen2.py`

Relevant RADLADS surface:

- `RWKV7State` carries both `layer_kv_states` and `layer_shift_states`.
- The model wraps RWKV7 attention in a Qwen2-like decoder layer with
  `Qwen2RMSNorm`, `Qwen2MLP`, residual paths, embedding, final norm, and LM
  model structure.
- RoPE is implemented through `Qwen2RotaryEmbedding` and
  `apply_rotary_pos_emb`, gated by config.
- Attention uses `q_proj`, `k_proj`, `v_proj`, `o_proj`, with
  `num_key_value_heads` and grouped KV expansion.
- Time/value/gate math uses RADLADS parameter names such as `w0/w1/w2`,
  `a0/a1/a2`, `v0/v1/v2`, `g1/g2` or `gate`, plus `k_k`, `k_a`, `r_k`, and
  optional `ln_x` group norm.
- The training path calls `chunk_rwkv7`; inference calls
  `fused_recurrent_rwkv7`; separate Triton/CUDA files contain recurrent/chunked
  kernel implementations.

## Comparison Table

| Area | RADLADS | QRWKV-XLA Current | Status | Priority | Notes |
| --- | --- | --- | --- | --- | --- |
| Backend registration | PyTorch `RWKV7Qwen2*` model classes | JAX student factory exposes `rwkv7_radlads_reference` | implemented | documentation-only | QRWKV-XLA registration is for distill/student smokes, not HF model compatibility. |
| State shape | KV state per layer shaped `[B,H,N,N]` plus shift state | KV matrix state per layer shaped `[B,H,N,N]`, exposed as `[L,B,H,N,N]` | partially implemented | P32 must-fix before kernel | Matrix state exists. Shift state and HF cache object semantics are missing. |
| Shift-state cache | `RWKV7State.layer_shift_states`, optional token shift machinery | No shift state; `step` only carries matrix state | missing | P32 must-fix before kernel | Needed before matching RADLADS cached decode behavior or parameterized token shift. |
| Decay semantics | `log_w = -exp(-0.5 - softplus(-w_lora))`; kernels use equivalent `exp(-exp(.))` forms | `log_w = -exp(-0.5) * sigmoid(w)` then `decay = exp(log_w)` | implemented | P32 must-fix before kernel | Algebraically aligned for the decay transform, but current `w` source is not RADLADS LoRA. |
| WKV matrix recurrence | Fused/chunk recurrent RWKV7 over `r, log_w, k, v, -kk, kk*a` | Scan recurrence with decay, `vk`, normalized `kk`, and `prev_state @ ab` | partially implemented | P32 must-fix before kernel | Shape is aligned. Exact `k` balancing, `a/b` parameterization, and output post-processing are not. |
| In-context update terms | Uses normalized `kk`; passes `-kk` and `kk*a` into recurrent op | Uses `ab = outer(-kk, kk * a + b)` with separate learned `wb` | partially implemented | P32 must-fix before kernel | Current `wb` is conceptually shaped like a second update vector but does not match current RADLADS modeling names or call site. |
| Query/key/value/output projections | `q_proj`, `k_proj`, `v_proj`, `o_proj` | `wr`, `wk`, `wv`, `wo` | partially implemented | P32 must-fix before kernel | Projection roles exist, but names, biases, grouped KV shapes, and checkpoint mapping do not match. |
| LoRA-rank decay path | `w0`, `w1`, `w2` with `tanh(xw @ w1) @ w2` | Dense `ww` projection plus `time_bias` | missing | P32 must-fix before kernel | Dense projection can train, but cannot load or compare RADLADS weights directly. |
| LoRA-rank ICLR path | `a0`, `a1`, `a2` | Dense `wa` projection | missing | P32 must-fix before kernel | Must be represented before any parameter mapping claim. |
| Value residual mix | `v0`, `v1`, `v2`; first layer sets `v_first`, later layers mix toward it | No `v_first` path or value residual mix | missing | P32 must-fix before kernel | This is core block math, not a scale-only concern. |
| Gate variants | `gate` linear or low-rank `g1/g2`, selected by `gate_rank_type` | Dense `wg` then sigmoid | partially implemented | P33 near-term quality/compatibility | A gate exists, but the low-rank and config variants do not. |
| `k_k`, `k_a`, `r_k` terms | Parameters exist; balance-state path changes `k`; `r_k` residual is present but commented in observed source | No `k_k`, `k_a`, or `r_k` parameters | missing | P33 near-term quality/compatibility | `k_k/k_a` matter when `balance_state` is false. `r_k` appears inactive in inspected modeling source. |
| Group norm / output scaling | Optional `ln_x` group norm; else scale by `N ** -0.5`; then `o_proj(x * g)` | Gates recurrent readout, applies `wo`, then `tanh(token + mixer)` | partially implemented | P32 must-fix before kernel | Current tanh residual is not Qwen/RADLADS attention output semantics. |
| Qwen decoder block | RMSNorm before attention, residual, RMSNorm before MLP, residual | No RMSNorm/MLP block; recurrent layer directly emits hidden states | missing | P32 must-fix before kernel | This is the largest model-compatibility gap. |
| Embedding/final norm/model wrapper | HF-style embedding, layer stack, final norm, CausalLM head | Student embedding and optional LM head only | partially implemented | P33 near-term quality/compatibility | Enough for distill smokes, not checkpoint/model parity. |
| RoPE | `Qwen2RotaryEmbedding` and `apply_rotary_pos_emb` when `use_rope` | No RoPE | missing | P32 must-fix before kernel | Should be in the slow reference before parity comparison, even if disabled by some configs. |
| Grouped KV heads | `num_key_value_heads`, repeated into attention heads | Single `num_heads`; no separate KV heads | missing | P32 must-fix before kernel | Needed for config compatibility and real checkpoint shape mapping. |
| Attention mask | Padding mask accepted as `[B,T]`; value stream masked for left padding | Padding mask accepted as `[B,S]`; value stream and outputs/mixers masked | partially implemented | P33 near-term quality/compatibility | Broad behavior is close, but exact cached/prefill semantics are unproven. |
| Cache update | Updates `RWKV7State` only when not training, `use_cache`, and cache exists | `step` returns final matrix state for `[B,1]` tokens | partially implemented | P32 must-fix before kernel | Step equivalence is covered, but object cache semantics and shift state are absent. |
| Training/inference kernel split | `chunk_rwkv7` for training, `fused_recurrent_rwkv7` for inference, plus Triton/CUDA paths | Pure JAX scan only | intentionally deferred | scale-only concern | Correct to defer until the slow math target is trustworthy. |
| Pallas/TPU kernel | Not the inspected RADLADS implementation; RADLADS has Triton/CUDA | None | intentionally deferred | scale-only concern | Out of scope for P31/P32 until math/block gaps close. |
| Distill hidden loss | Not the RADLADS model concern | Hidden MSE is wired and tested | implemented | documentation-only | Proves QRWKV-XLA training plumbing, not RADLADS parity. |
| Distill logits KL | Not the RADLADS model concern | Optional student logits and logits KL are wired and tested | implemented | documentation-only | Proves target-loss plumbing only. |
| Target bundle contract | Not RADLADS-native | Manifest plus NPZ shards for hidden/logits/mixer targets | implemented | documentation-only | Useful project infrastructure, not model math. |
| Parameter mapping compatibility | Native PyTorch names and shapes | No mapping layer from RADLADS names to JAX params | missing | P32 must-fix before kernel | Required before numerical parity or checkpoint import claims. |
| Numerical parity tests | PyTorch/kernel outputs define reference behavior | No comparison against RADLADS outputs | missing | P32 must-fix before kernel | Add tiny source-confirmed fixtures only after slow reference has compatible parameter surface. |
| Export / checkpoint compatibility | HF `PreTrainedModel` conventions | Local JSON + NPZ checkpoints for QRWKV-XLA students | missing | export-only concern | Should wait until parameter mapping is concrete. |
| Quality evaluation | Downstream task/model quality outside these files | No quality claim | intentionally deferred | not needed | Not relevant until real model compatibility exists. |

## Must-Fix Before Kernel Work

- Replace the current direct recurrent layer shell with a slow JAX compatibility
  target that can represent the RADLADS Qwen decoder block contract:
  RMSNorm-attention-residual-RMSNorm-MLP-residual.
- Add an explicit state contract that includes both layer KV matrix state and
  layer shift state, even if token shift remains disabled by default.
- Add grouped KV head configuration and expansion semantics.
- Add RoPE support compatible with RADLADS/Qwen config fields.
- Add RADLADS parameter naming and shape compatibility for projections,
  `w0/w1/w2`, `a0/a1/a2`, `v0/v1/v2`, gate variants, and relevant
  `k_k/k_a/r_k/ln_x` fields.
- Remove or isolate the current `tanh(token + mixer)` behavior from any path
  that claims RADLADS block compatibility.
- Add tiny equation-level tests for the slow JAX path, including full-sequence
  vs step cache equivalence after the shift-state contract exists.
- Add a minimal parameter mapping audit fixture before claiming numerical
  parity against RADLADS outputs.

## Can Wait Until Export / Scale

- HF `PreTrainedModel` save/load compatibility.
- Real RADLADS checkpoint import/export.
- Orbax or sharded checkpoint formats.
- Large teacher target storage formats.
- `pjit`, parameter sharding, model parallelism, multi-host concerns.
- Pallas/TPU optimized recurrence kernels.
- Triton/CUDA comparison at large sequence/head sizes.

## Intentionally Deferred / Not Needed

- CUDA or Triton implementation inside QRWKV-XLA.
- Model quality evaluation or benchmark claims.
- Real Qwen training runs as default validation.
- Full HF generation API compatibility in the slow reference phase.
- Broad refactors of existing distill, target bundle, optimizer, tokenizer, or
  tracking infrastructure.

## Recommended P32 Scope

P32 should be a math-completion and block-compatibility phase, not kernel work.

Concrete P32 scope:

- Define a JAX RADLADS block config that mirrors the relevant
  `RWKV7Qwen2Config` fields for small models.
- Implement the slow Qwen-style decoder layer shell around the current
  recurrence: RMSNorm, RWKV7 attention/mixer output, residual, RMSNorm, MLP,
  residual.
- Add the state dataclass/tree for `[layers, batch, heads, head_size,
  head_size]` KV state plus per-layer shift state.
- Add grouped KV head shapes and repeat semantics.
- Add optional RoPE for `r` and `k`.
- Add RADLADS-named parameter initialization/mapping stubs with explicit
  unsupported-field errors rather than silent shape drift.
- Keep tests tiny: shape checks, config validation, full-vs-step state
  equivalence, RoPE on/off smoke, grouped-KV shape smoke, and a one-step
  distill smoke if the public factory surface changes.

P32 should end with a clearer statement of whether P33 can start numerical
parity fixtures or whether more slow-reference math remains.

## Recommended P33+ Scope

- P33: tiny numerical comparison fixtures against RADLADS PyTorch for the slow
  reference, once parameter names and block shapes are compatible.
- P34: parameter import/export compatibility for a tiny synthetic RADLADS
  checkpoint.
- P35+: optimized recurrence work only after slow-reference tests can catch
  regressions in recurrence math, cache behavior, RoPE, grouped KV, and block
  outputs.
- Later: Pallas/TPU, `pjit`, sharding, and export hardening.

## Do-Not-Claim Boundaries

Do not claim:

- full RADLADS parity
- checkpoint compatibility with RADLADS weights
- numerical parity with RADLADS PyTorch, Triton, or CUDA outputs
- Qwen decoder block parity
- RoPE parity
- grouped KV parity
- cached inference parity
- TPU kernel readiness
- model quality

Allowed wording remains: RADLADS-shaped JAX reference backend, not full
RADLADS parity.
