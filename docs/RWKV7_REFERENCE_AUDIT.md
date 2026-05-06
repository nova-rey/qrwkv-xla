# RWKV7 Reference Audit

`rwkv7_reference` remains the original lightweight smoke backend. It is useful
for exercising QRWKV-XLA training, checkpointing, generation, and distributed
plumbing, but it is not RADLADS RWKV7 math parity.

`rwkv7_radlads_reference` is the newer partial RADLADS-aligned backend. It adds
head-wise matrix recurrent state, normalized key direction, RADLADS-style decay,
in-context matrix update terms, a final-state API, and token-step equivalence
coverage. It is intentionally slow and readable.

Current limits:

- no RADLADS checkpoint import or parameter-name compatibility
- no Qwen decoder block parity, RoPE, group norm, grouped KV heads, value
  residual mixing, or RADLADS LoRA time-mix modules
- no CUDA, Triton, Pallas, or production kernel path
- no numerical tolerance claim against RADLADS torch outputs

Masking note: `rwkv7_radlads_reference` treats `attention_mask` as padding. The
value stream is zeroed for masked tokens, while the recurrence still applies
decay and in-context update math. This follows the RADLADS left-padding
behavior more closely than freezing the recurrent state.
