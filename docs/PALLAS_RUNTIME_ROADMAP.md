# Pallas Runtime Roadmap

The Pallas runtime remains opt-in. The trusted default WKV runtime is
`reference` until later phases prove broader correctness and integration.

## Runway

- P81: opt-in selector.
- P82: minimal Pallas execution probe.
- P83: tiny reference-vs-Pallas parity.
- P84: broader one-step shape/dtype parity.
- P85: sequence/scan-style Pallas WKV parity.
- P86: fixture-family opt-in integration.
- P87: TPU compile/performance smoke.

## Current Scope

P84 compares the interpreted Pallas one-step update against:

```text
state * decay[..., None, :] + k[..., :, None] * v[..., None, :]
```

The matrix is intentionally small: required float32 B/H/D cases plus optional
bfloat16 rows. Passing P84 does not promote Pallas as the default, prove
sequence parity, prove TPU performance, or integrate Pallas into model fixture
runs.
