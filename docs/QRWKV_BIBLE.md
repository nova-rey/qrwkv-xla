# QRWKV-XLA Bible

This file is append-only. New phases and decisions should be appended below rather than rewriting earlier history.

## Phase 0 / Prompt A — Project Birth

QRWKV-XLA begins as a JAX/XLA-first reimplementation of the RADLADS-style recurrent conversion idea.

The existing RADLADS repository remains the reference implementation and design ancestor, but this project is not a port. The original repo includes GPU-oriented dependencies and CUDA/Triton paths, so QRWKV-XLA will be built with XLA and TPU constraints as first-class design requirements.

Initial target:
- Teacher: Qwen3.latest policy, with Qwen3.0 fallback
- Student: RWKV7-style recurrent architecture
- Runtime: CPU debug first, TPU smoke second, TPU scale later
- Workflow: Nova design/specs, Nyx/Codex implementation
