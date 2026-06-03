# Runtime Environment Preflight

P109 adds read-only runtime environment hygiene for burn-readiness work.

## Purpose

P109 is infrastructure hygiene. It inspects whether the current Python/JAX
runtime is sane enough to attempt later TPU burn-readiness work. It is not
model behavior, training, benchmarking, pjit sharding, or Pallas promotion.

## JAX / TPU Inspection

The preflight lazily inspects JAX when available:

- Python version
- JAX and JAXLIB versions
- default JAX backend
- visible JAX devices
- TPU platform/device-kind detection

If JAX import or device inspection fails, the report records that condition
without crashing. Baseline tests do not require TPU/GPU or a successful JAX
import.

## Transparent Hugepages

The preflight reads:

```text
/sys/kernel/mm/transparent_hugepage/enabled
```

It classifies transparent hugepages as:

- `enabled`: active mode is `[always]`
- `disabled`: active mode is `[madvise]` or `[never]`
- `unavailable`: the sysfs file cannot be read
- `unknown`: the active mode cannot be parsed

Transparent hugepages warnings are environment/runtime readiness signals, not
model behavior.

## Read-Only Default

The preflight is read-only by default. It never runs `sudo` automatically and
does not mutate `/sys` unless explicitly requested.

The report always includes the human command for enabling transparent
hugepages:

```bash
sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"
```

## Commands

Read-only JSON report:

```bash
python scripts/run_runtime_environment_preflight.py \
  --output artifacts/p109_runtime_environment/runtime_environment_report.json
```

Require TPU:

```bash
python scripts/run_runtime_environment_preflight.py \
  --output artifacts/p109_runtime_environment/runtime_environment_report.json \
  --require-tpu
```

Explicitly request transparent hugepage mutation:

```bash
python scripts/run_runtime_environment_preflight.py \
  --output artifacts/p109_runtime_environment/runtime_environment_report.json \
  --enable-transparent-hugepages
```

The mutation path writes `always` directly to the configured sysfs path if
permitted. It does not invoke `sudo`.

## Report Fields

The report includes:

- `phase`: `P109`
- `status`: `pass`, `warn`, or `fail`
- `python_version`
- `jax_available`, `jax_version`, `jaxlib_version`, and `default_backend`
- `devices` with `id`, `platform`, and `device_kind`
- `tpu_devices_detected`
- `transparent_hugepages`
- `mutation_attempted` and `mutation_ok`
- `claims_not_made`

## What P109 Proves

P109 proves the repo has a CPU-safe, JSON-producing preflight for JAX/TPU
visibility and transparent hugepage readiness.

## What P109 Does Not Prove

P109 does not prove training readiness, performance, pjit readiness, Pallas
default readiness, big-burn readiness, model quality, or TPU availability.

## Future Phases

P110 is the next planned checkpoint: Mini Eval Harness Smoke.
