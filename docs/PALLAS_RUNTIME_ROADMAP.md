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

P87 integrates the opt-in Pallas WKV path into the existing tiny WKV7
fixture-family correctness harness. The reference runtime remains the default;
Pallas remains an explicit candidate/runtime request.

The P87 fixture-family path runs the P43 `tiny_wkv7_correctness` cases through
the Pallas-backed candidate and reports pass, fail, skipped, and unsupported
cases without changing fixture tensors or tolerances.

Passing P87 does not promote Pallas as the default, prove full-model parity,
prove TPU performance, prove real training throughput, or prove model quality.
