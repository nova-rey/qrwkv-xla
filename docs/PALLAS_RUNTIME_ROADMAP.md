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

P88 adds a tiny TPU compile/execution smoke harness for the opt-in Pallas WKV
path. The reference runtime remains the default; Pallas remains explicit.

CPU-only runs report `status=unavailable` with
`reason=no_tpu_devices_detected`. TPU runs attempt tiny JIT/lowering/compile,
execution, and a numeric check against the trusted reference path.

Passing P88 does not promote Pallas as the default, prove production readiness,
prove real training throughput, or prove model quality.
