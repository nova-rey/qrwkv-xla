# Offline Target Consumption

P95 adds the first offline target consumption smoke. The boundary loads a
validated `TeacherTargetStore` shard and exposes it as an `OfflineTargetBatch`
with `input_ids`, `attention_mask`, and `teacher_logits`.

The implementation lives in `src/qrwkv_xla/targets/consumption.py`:

- `OfflineTargetBatch`
- `load_offline_target_batch(store, shard_id=0)`
- `mse_logits_loss(student_logits, teacher_logits)`

The loader validates the store through the P93/P94 artifact contract, reads one
local `.npz` shard, checks student-facing shapes against metadata, and returns
NumPy arrays. The loss helper computes a finite scalar MSE for matching logits
shapes and raises `ValueError` on shape mismatch.

The smoke path uses `SyntheticTeacherBackend` to create the store, then runs the
real `CurrentQRWKVStudentBackend` with `emit_logits=True` and matching
vocabulary size. This proves that stored offline logits can be consumed by the
current student-side logits surface without a live teacher process.

P95 does not add optimizer steps, training loops, gradient updates, live
Hugging Face/Qwen calls, GPU/TPU requirements, trainer refactors, WKV math
changes, Pallas promotion, or fixture edits.
