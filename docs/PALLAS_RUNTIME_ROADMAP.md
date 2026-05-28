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
- P89: TPU smoke tracing-boundary cleanup.
- P90: record real TPU pass and close Pallas runway.

## Current Scope

P88 adds a tiny TPU compile/execution smoke harness for the opt-in Pallas WKV
path. The reference runtime remains the default; Pallas remains explicit.

CPU-only runs report `status=unavailable` with
`reason=no_tpu_devices_detected`. TPU runs attempt tiny JIT/lowering/compile,
execution, and a numeric check against the trusted reference path.

Passing P88 does not promote Pallas as the default, prove production readiness,
prove real training throughput, or prove model quality.

## Runway Closure

Pallas runtime runway closure:

- P87 fixture-family opt-in integration passed.
- P88 TPU smoke harness added.
- P89 fixed the tracing boundary that initially failed with
  `DynamicJaxprTracer has no attribute block_until_ready`.
- Real TPU v5 lite rerun passed:
  - `jit_lowering_ok=true`
  - `execution_ok=true`
  - `numeric_check_ok=true`
  - `max_abs_error=0.0`
- Pallas remains opt-in.
- Reference remains default.
- No production, training, throughput, full-model quality, or Pallas-default
  readiness claims are made.

The Pallas runway is complete enough to transition from toy feasibility and
integration smoke work into post-Pallas architecture extraction planning.
