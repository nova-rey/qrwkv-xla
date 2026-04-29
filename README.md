# QRWKV-XLA

QRWKV-XLA is a JAX/XLA-first recurrent conversion pipeline inspired by RADLADS.

It aims to distill Qwen-family transformer teachers into RWKV7-style recurrent
students using TPU-friendly training infrastructure.

## Current Status

Phase 1: target artifact store foundation.

The project can now define, write, read, validate, inspect, and test fake
teacher target bundles on CPU without requiring JAX, PyTorch, TPU, GPU, or
network access.

## Design Principles

- Full-system architecture from day one
- Tiny configs, not disposable toy systems
- JAX/XLA-first student training
- PyTorch/Hugging Face teacher extraction later
- CPU local development
- TPU smoke tests when available
- No CUDA/Triton dependency in student training path
- Simple, inspectable artifact formats first

## Reference

This project uses `nova-rey/radlads-TPU-adapter` as a conceptual and
architectural reference, not as code to directly port.

The reference RADLADS lineage includes RAD-RWKV6/RAD-RWKV7 components,
Hugging Face conversion scripts, staged configs, Lightning trainer flows,
`lm_eval` support, and inference support. QRWKV-XLA is being rebuilt around XLA
and TPU constraints from day one instead of carrying over GPU-shaped internals.
