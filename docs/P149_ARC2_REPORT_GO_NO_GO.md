# P149 Arc 2 Report / Go-No-Go

P149 closes the real student integration and teacher-pure capture arc with a
deterministic report generator.

The report reads `docs/QRWKV_SNAPSHOT.yaml`, verifies the P140-P148 evidence
flags, and writes:

- `p149_arc2_report.json`
- `p149_arc2_summary.md`

## CLI

```bash
python scripts/run_fingerprint_arc2_report.py \
  --output-dir /tmp/qrwkv_p149_arc2 \
  --snapshot docs/QRWKV_SNAPSHOT.yaml \
  --overwrite
```

## Recommendation

When the required P140-P148 evidence is present, P149 emits:

```text
recommendation: go_with_constraints
```

This means the next work may proceed to larger controlled fingerprint
experiments under explicit gates. It does not mean production readiness, scale
readiness, RADLADS parity, or a general quality result.

## Constraints

The report keeps these gates visible:

- Add a trained non-fingerprint baseline before method-vs-method claims.
- Add held-out evaluation before stronger quality-per-byte language.
- Keep teacher, corpus, student, and artifact byte budgets fixed and reported.
- Keep claims scoped to measured tiny or controlled settings.
- Do not claim production readiness, scale readiness, RADLADS parity, or Pallas
  default readiness.

## Open Gaps

P149 records unresolved gaps that should shape the next arc:

- no competitive trained non-fingerprint baseline yet
- P148 uses train-artifact reuse rather than held-out evaluation
- artifacts and runs remain tiny CPU-safe smokes
- main runner still lacks exemplar/mixed-objective training
