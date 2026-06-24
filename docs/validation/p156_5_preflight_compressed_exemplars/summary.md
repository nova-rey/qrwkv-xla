# P156.5 Compressed Exemplar Payload Wiring

Status: implementation complete; exact production smoke unavailable in this workspace.

Implemented:

- Real-teacher capture defaults to `cascaded_soft_labels_v1` exemplars with fixed `top_k=256`.
- The fingerprint capture manifest records the full encoding contract.
- Compressed exemplar rows omit `teacher_probs` and shard output.
- The reservoir encodes only candidates admitted to the bounded retained set.
- The validator accepts legacy dense rows and fails closed on malformed compressed rows.
- The loader and exemplar loss consume compressed rows directly, without dense reconstruction.
- The shared cascaded loss now avoids host boolean conversion when traced.

Validation:

- `.venv/bin/ruff check scripts/build_real_teacher_fingerprint_artifact.py src/qrwkv_xla tests/test_compressed_fingerprint_exemplars.py tests/test_tiny_real_teacher_fingerprint_capture.py`
- `.venv/bin/pytest -q tests/test_fingerprint_capture.py tests/test_fingerprint_exemplars.py tests/test_compressed_fingerprint_exemplars.py tests/test_tiny_real_teacher_fingerprint_capture.py tests/test_cascaded_soft_labels_textbook.py tests/test_topk_tail_textbook.py`
- Result: `69 passed, 1 skipped`

Production smoke status:

- The exact 25-example Cardboard probe was attempted with local-files-only HF teacher settings.
- It failed fast with `status=unavailable` because `transformers` is not installed.
- A local cache check also found no `sshleifer/tiny-gpt2` cache.
- The exact Cardboard corpus/tranche was not found locally.
- The 25-example and 1,000-example production receipts are therefore recorded as unavailable, not substituted with synthetic data.
