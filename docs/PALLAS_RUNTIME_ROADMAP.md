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

P86 compares a short sequence reference recurrence against a scan-style Pallas
scaffold:

```text
state * decay[..., None, :] + k[..., :, None] * v[..., None, :]
```

The current P86 scaffold is labeled `jax_scan_pallas_step_scaffold`: it uses
`jax.lax.scan` over the interpreted one-step Pallas update, compares final and
per-step states against the trusted reference sequence loop, and records an
explicit `scan_scaffold_pass` status only when required cases pass. Passing P86
does not promote Pallas as the default, prove fixture-family/full-model parity,
prove TPU performance, or prove real training throughput.
