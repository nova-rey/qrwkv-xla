# Second StudentBackend Smoke

P102 closes the current modularity arc by adding a tiny/debug second
`StudentBackend`. This backend is a socket-test backend, not a production model.

## Purpose

P102 proves the student architecture registry can select more than the current
QRWKV backend while preserving the separation between teacher vocab contract,
student architecture, and runtime.

## TinyDebugStudentBackend

`TinyDebugStudentBackend` lives in
`src/qrwkv_xla/students/tiny_debug_backend.py`.

It produces deterministic finite logits shaped `[B,T,V]`, where `V` comes from
the `VocabContract`. The logits are simple arithmetic functions of `input_ids`
and vocab index. The backend has a minimal state counter and supports
export/import round trips.

This backend is intentionally not useful as a language model. It exists to test
the registry/socket.

## Registry Selection

The registry now exposes:

- `current_qrwkv`
- `tiny_debug`

`architecture_id=None` still defaults to `current_qrwkv`.

## Vocab Contract Handling

`tiny_debug` is born from the same `VocabContract` input as other student
backends. Different vocab sizes produce logits with matching last dimensions.

## Runtime Policy

`tiny_debug` supports the reference/default path. Pallas requests fail clearly
because this debug backend has no Pallas implementation. Pallas remains opt-in
for supported backends and is not promoted.

## Compatibility/Loss Smoke

P102 proves a compatible target artifact can pass the P99 gate, load through
the P95 offline target path, run through `TinyDebugStudentBackend`, and compute
a finite MSE logits loss.

## What P102 Proves

P102 proves the registry/socket can select a second backend and route it through
existing contract/loss gates.

## What P102 Does Not Prove

P102 does not prove a production second architecture, model quality, training
readiness, Pallas support for `tiny_debug`, Qwen-specific support, tokenizer
remapping, or optimized runtime behavior.

## Next Arc

The next arc can start real-teacher rehearsal and burn readiness work, beginning
with a tiny real-teacher overfit rehearsal.
