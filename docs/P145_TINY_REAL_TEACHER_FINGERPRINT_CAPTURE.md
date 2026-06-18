# P145 Tiny Real Teacher Fingerprint Capture

P145 is the first tiny real-teacher producer-side fingerprint capture. It sends
logits from a tiny local-files-only HF causal LM teacher through the calibrated
P143/P144 capture pipeline and emits a valid `behavioral_fingerprint` artifact.

This is a capture smoke, not a benchmark or a student quality result.

## Path

The P145 path is:

```text
tiny local text fixture
-> HF causal LM teacher logits
-> FingerprintCaptureExample records
-> teacher_side_capture_skeleton_v0
-> behavioral_fingerprint artifact
-> validator / loaders / summary
-> consumer sanity
```

The dedicated CLI is:

```bash
python scripts/build_real_teacher_fingerprint_artifact.py \
  --teacher-model sshleifer/tiny-gpt2 \
  --tokenizer sshleifer/tiny-gpt2 \
  --texts tests/fixtures/fingerprint_capture_real_teacher/tiny_texts.jsonl \
  --output-dir /tmp/qrwkv_p145_tiny_real_teacher_fingerprint \
  --sequence-length 32 \
  --max-examples 4 \
  --max-target-positions 64 \
  --max-exemplars 16 \
  --bounds-method quantile \
  --exemplar-selection-policy stratified_interestingness_v0 \
  --local-files-only \
  --overwrite
```

Tests do not download models. If the tiny teacher is not available in the local
HF cache, the optional CLI smoke skips cleanly.

## Output

P145 writes:

```text
manifest.json
modes.json
targets/targets-00000.jsonl
exemplars/exemplars-00000.jsonl
capture_summary.json
```

The manifest records real teacher metadata including backend, model path,
tokenizer path, local-files-only mode, vocab size, dtype, and CPU device. The
summary distinguishes:

- `phase: P145`
- `run_kind: tiny_real_teacher_capture`
- `capture_engine: teacher_side_capture_skeleton_v0`

It also records examples, tokens, target positions, modes discovered, records
per mode, bounds method, exemplar policy, max exemplars, exemplars retained,
artifact validation, loader status, and consumer sanity.

## Consumer Sanity

P145 tries the strongest cheap consumer check:

1. P141 one-step `fingerprint_corridor`
2. P140 real-student forward smoke
3. loader-only fallback with an explicit reason

Large HF vocabularies can make cheap CPU student instantiation inappropriate.
In that case the summary records `consumer_sanity.kind = loader_only` and an
explicit reason. No vocab shrinkage or remapping is performed.

## Non-Claims

P145 does not prove real-scale teacher capture, TOME/textbook integration,
artifact convergence, student quality improvement, baseline comparison,
quality-per-byte gain, or production capture performance.

P146 should use a tiny real-teacher fingerprint artifact for a real student
training rehearsal.
