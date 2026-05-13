# P52 — RADLADS Tiny Fixture Parameter Cleanup

## Overview

P52 audits and cleans RADLADS tiny fixture parameter initialization after P51 found non-finite/extreme source payload values.

It traces parameters from RADLADS live module to pre-save dictionary to saved NPZ to QRWKV import, adds finite payload validation and deterministic finite fixture generation where needed, regenerates clean fixtures, and reruns replay diagnostics/comparison.

It does **not** implement Pallas kernel, prove training throughput, or claim full RADLADS parity unless source initialization is also validated clean.

## What P51 Found

P51 discovered that the RADLADS tiny fixture parameter payload contained suspicious/non-finite/extreme values:

- **Non-finite cases**: 24/24 before P51, 20/24 after P51 stabilization
- **First non-finite surface** (all-math case): `layers.0.self_attn.w1_projection` at stage `w1_projection`
- **Suspicious surfaces**: w1/w2, g1/g2, v1/v2, a0, w0, r_k
- **Largest absolute values**: around ~1.79e35

These values may indicate:
- Uninitialized tensors (`np.empty`/`torch.empty`/`jax.numpy.empty`)
- Serialization corruption (views without `.copy()`)
- Import/layout corruption (wrong dtype/layout)
- Valid RADLADS initialization interpreted incorrectly
- Or a real QRWKV-XLA formula/layout bug

## Why P52 Blocks Progress

The project should not proceed to Pallas/kernel work while the slow reference replay path is fed by suspicious or non-finite RADLADS fixture parameters.

Before debugging deeper numerical mismatch, P52 must make the test specimen clean.

## Audit Commands

### Provenance Audit

Trace parameters from RADLADS live module through save/import stages:

```bash
python scripts/audit_radlads_fixture_parameter_provenance.py \
 --radlads-repo /home/nyx/.openclaw/workspace/_refs/RADLADS \
 --manifest artifacts/p49_radlads_numerical_parity/radlads_fixtures/manifest.json \
 --parameters artifacts/p49_radlads_numerical_parity/radlads_fixtures/radlads_parameters.npz \
 --out artifacts/p52_radlads_fixture_parameter_cleanup \
 --overwrite
```

Output:
- `artifacts/p52_radlads_fixture_parameter_cleanup/parameter_provenance_report.json`
- `artifacts/p52_radlads_fixture_parameter_cleanup/P52_PARAMETER_PROVENANCE.md`

### Parameter Payload Validation

Validate that all parameters are finite and below extreme threshold:

```bash
python scripts/validate_radlads_parameter_payload.py \
 --parameters artifacts/p49_radlads_numerical_parity/radlads_fixtures/radlads_parameters.npz \
 --out artifacts/p52_radlads_fixture_parameter_cleanup/validation \
 --overwrite \
 --extreme-threshold 1e6
```

Output:
- `artifacts/p52_radlads_fixture_parameter_cleanup/validation/parameter_audit_report.json`
- `artifacts/p52_radlads_fixture_parameter_cleanup/validation/P52_PARAMETER_AUDIT.md`

## Audit Stages

Parameters are traced across four stages:

1. **radlads_live**: Live RADLADS module parameters after initialization
2. **pre_save**: Transformed/exported RADLADS parameter dict before NPZ save
3. **saved_npz**: Saved `radlads_parameters.npz`
4. **qrwkv_imported**: QRWKV-XLA loaded/imported parameter tree

For each parameter, audit reports:
- `name`, `stage`, `shape`, `dtype`
- `min`, `max`, `mean`, `std`, `abs_max`
- `finite_count`, `nan_count`, `posinf_count`, `neginf_count`
- `all_zero`, `all_same`, `sha256` (stable hash)
- `status`: `finite_ok`, `non_finite`, `extreme_value`, `shape_mismatch`, `dtype_mismatch`, `changed_between_stages`, `missing`, `defaulted`, `excluded`, `unsupported`

## Init Policies

### radlads_source

Uses actual RADLADS initialization path (if available).

**Use when**: RADLADS source initialization is proven clean and finite.

### deterministic_finite

Deterministic small finite values for parity plumbing/debugging.

**Use when**: `radlads_source` mode remains poisoned or is unavailable.

## Validation Thresholds

- **Default extreme threshold**: `1e6`
- **Extreme value**: `abs(value) > extreme_threshold`
- **Non-finite**: `NaN`, `+inf`, or `-inf`

Do not silently accept `1e35`-scale values unless RADLADS source proves this is intentional.

## Regenerate Clean Fixtures

After audit identifies the issue, regenerate clean fixtures:

```bash
python scripts/generate_radlads_tiny_numerical_fixtures.py \
 --radlads-source /home/nyx/.openclaw/workspace/_refs/RADLADS \
 --out artifacts/p52_radlads_fixture_parameter_cleanup/radlads_fixtures_clean \
 --seed 5050 \
 --overwrite
```

If the RADLADS source initialization remains poisoned or unavailable, generate a
clean deterministic parameter payload instead:

```bash
python scripts/generate_radlads_tiny_numerical_fixtures.py \
 --init-policy deterministic_finite \
 --out artifacts/p52_radlads_fixture_parameter_cleanup/radlads_fixtures_clean \
 --seed 5050 \
 --overwrite
```

The deterministic path writes `radlads_parameters.npz` with small finite values
and records `parameter_payload_init_policy`,
`parameter_payload_source`, and `parameter_payload_validation` in the manifest.

## Replay Diagnostics

After clean fixtures are generated, rerun replay diagnostics:

```bash
python scripts/diagnose_radlads_replay_nonfinite.py \
 --manifest artifacts/p52_radlads_fixture_parameter_cleanup/radlads_fixtures_clean/manifest.json \
 --out artifacts/p52_radlads_fixture_parameter_cleanup/replay_clean \
 --all-cases \
 --overwrite
```

## Interpreting Outcomes

### Best Result

```text
all clean fixtures replay finitely
at least some surfaces pass allclose
remaining failures are numeric parity/layout/formula mismatches
```

### Good Result

```text
all or most clean fixtures replay finitely
no surfaces pass yet
failures are finite numeric errors
```

### Acceptable Diagnostic Result

```text
clean parameters are proven finite
replay still becomes non-finite
first non-finite now points to a QRWKV-XLA formula/layout bug
```

### Not Acceptable

```text
source parameters remain non-finite without exact source-backed explanation
generator accepts poisoned fixtures silently
replay failures hidden or counted as pass
```

## Likely Fix Patterns

### Case A — Uninitialized Tensor Bug

**Symptom**: RADLADS live or pre-save parameter has `1e35`/NaN/Inf.

**Fix**: Call proper initializer, replace empty tensors with zeros/small deterministic values.

### Case B — Serialization Corruption

**Symptom**: RADLADS live/pre-save is finite, saved NPZ is poisoned.

**Fix**: Use `.detach().cpu().contiguous().copy()` before numpy save.

### Case C — Import/Layout Corruption

**Symptom**: Saved NPZ is finite, QRWKV imported tree is poisoned.

**Fix**: Fix parameter import dtype/layout/transpose, add shape/hash checks.

### Case D — Valid Source Extreme Values

**Symptom**: RADLADS source intentionally initializes extreme values.

**Fix**: Trace RADLADS runtime transform, ensure QRWKV applies identical transform, document source lines.

### Case E — Clean Parameters but Replay Explodes

**Symptom**: All parameters finite and small, replay first non-finite still appears.

**Fix**: Return to P51-style math diagnostics, likely formula/layout/sign/activation bug.

## Testing

Run tests:

```bash
cd /home/nyx/.openclaw/workspace/qrwkv-xla
.venv/bin/python -m pytest tests/test_radlads_parameter_provenance.py -v
.venv/bin/python -m pytest tests/test_radlads_replay_diagnostics.py tests/test_radlads_parameter_replay.py -v
.venv/bin/python -m pytest -v
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

## Timeline Docs

P52 audit was appended to:

- `docs/QRWKV_BIBLE.md`
- `docs/QRWKV_SNAPSHOT.yaml`
- `docs/PHASE_CHECKLIST.md`

## Known Caveats

- **P52 does not implement Pallas**
- **P52 does not prove full RADLADS parity** unless all compared surfaces pass
- **P52 does not prove training throughput**
- **P52 does not prove model quality**
- **Deterministic finite fixtures are a replay/plumbing stabilizer**, not a complete proof of source-initialized RADLADS parity unless source init is also validated clean.

## Files Changed

- `scripts/audit_radlads_fixture_parameter_provenance.py` — CLI for parameter provenance audit
- `scripts/validate_radlads_parameter_payload.py` — CLI for parameter payload validation
- `src/qrwkv_xla/parity/audit_radlads_parameter_provenance.py` — Core audit logic
- `src/qrwkv_xla/parity/radlads_fixture_validation.py` — Finite parameter validation
- `tests/test_radlads_parameter_provenance.py` — Tests for audit functionality
- `docs/RADLADS_TINY_FIXTURE_PARAMETER_CLEANUP.md` — This documentation

## Command Summary

```bash
# 1. Audit parameter provenance
python scripts/audit_radlads_fixture_parameter_provenance.py

# 2. Validate parameter payload
python scripts/validate_radlads_parameter_payload.py

# 3. Regenerate clean fixtures (if source is clean)
python scripts/generate_radlads_tiny_numerical_fixtures.py

# 4. Regenerate clean fixtures (if source is poisoned)
python scripts/generate_radlads_tiny_numerical_fixtures.py --init-policy deterministic_finite

# 5. Rerun replay diagnostics
python scripts/diagnose_radlads_replay_nonfinite.py --all-cases

# 6. Run tests
.venv/bin/python -m pytest tests/test_radlads_parameter_provenance.py -v
.venv/bin/python -m pytest -v
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

## Acceptance Criteria

P52 is accepted when:

- [ ] Parameter provenance audit script exists
- [ ] Provenance report traces saved NPZ → QRWKV import (at minimum)
- [ ] Fixture generator initialization path is audited
- [ ] Generator no longer silently accepts non-finite parameter payloads
- [ ] Manifest records init policy and parameter validation status
- [ ] Clean fixture set is generated under P52 artifact path
- [ ] Clean `radlads_parameters.npz` has zero non-finite values
- [ ] Extreme values are removed or source-backed/documented
- [ ] QRWKV imported parameters match saved NPZ for mapped values
- [ ] Replay is rerun against clean fixtures
- [ ] Replay result is finite for at least one case, preferably all
- [ ] Remaining failures are reported as finite numeric errors or explicit unsupported
- [ ] Docs explain audit/generation/replay commands and caveats
- [ ] Tests cover validation, provenance, fixture generation, and reports
- [ ] P51/P50/P49 tests remain intact
- [ ] Full local tests pass
- [ ] Ruff passes
- [ ] Ruff format passes
