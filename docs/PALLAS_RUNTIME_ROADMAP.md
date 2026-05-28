# Pallas Runtime Roadmap

The Pallas runtime remains opt-in. The trusted default WKV runtime is
`reference` until later phases prove broader correctness and integration.

## Runway

- P81: opt-in selector.
- P82: minimal Pallas execution probe.
- P83: tiny reference-vs-Pallas parity.
- P84: broader one-step shape/dtype parity.
- P85: short-sequence repeated one-step Pallas WKV parity.
- P86: fused/scan Pallas WKV kernel scaffold.
- P87: fixture-family opt-in integration.
- P88: TPU compile/performance smoke.

## Current Scope

P85 compares a short sequence reference recurrence against repeated calls to
the interpreted Pallas one-step update:

```text
state * decay[..., None, :] + k[..., :, None] * v[..., None, :]
```

The matrix is intentionally small: required float32 B/H/D/T short-sequence
cases plus optional bfloat16 rows. Passing P85 does not promote Pallas as the
default, implement or prove a fused scan kernel, prove TPU performance, or
integrate Pallas into model fixture runs.
