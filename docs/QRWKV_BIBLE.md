# QRWKV-XLA Bible

This file is append-only. New phases and decisions should be appended below rather than rewriting earlier history.

## Phase 0 / Prompt A — Project Birth

QRWKV-XLA begins as a JAX/XLA-first reimplementation of the RADLADS-style recurrent conversion idea.

The existing RADLADS repository remains the reference implementation and design ancestor, but this project is not a port. The original repo includes GPU-oriented dependencies and CUDA/Triton paths, so QRWKV-XLA will be built with XLA and TPU constraints as first-class design requirements.

Initial target:
- Teacher: Qwen3.latest policy, with Qwen3.0 fallback
- Student: RWKV7-style recurrent architecture
- Runtime: CPU debug first, TPU smoke second, TPU scale later
- Workflow: Nova design/specs, Nyx/Codex implementation

## Phase 0.5 — Foundation Normalization and Contracts

The initial scaffold existed but needed normalization into valid, readable
multiline files. This pass updates the workflow model from A/B/C prompts to Nyx
as the primary implementation agent with Codex as a sub-agent, then adds the
first config and target artifact contract layer.

## Phase 1 — Target Artifact Store Foundation

This phase adds the first durable data contract between future PyTorch/Hugging
Face teacher extraction and future JAX/XLA student training. The project can
now create fake teacher target bundles, write and read manifest JSON, validate
NPZ shards, inspect bundle metadata, and test the artifact store without
requiring GPU, TPU, PyTorch, JAX, or network access.

## Phase 2 — Teacher Exporter Interface + Fake Export Pipeline

This phase adds the exporter-shaped side of the pipeline. QRWKV-XLA now has a
teacher export configuration schema, export request/result contracts, a
TeacherExporter protocol, a deterministic fake exporter, and a CLI entrypoint
that writes valid target bundles through the artifact store. Real
Qwen/PyTorch/Hugging Face loading remains intentionally deferred.

## Phase 2.5 — Editable Install Validation and CI Stabilization

This stabilization phase keeps the P2 behavior intact while standardizing how
the repository is validated. QRWKV-XLA now treats `python -m pip install -e ".[dev]"`
as the blessed development setup, with scripts and tests importing the installed
package instead of using `PYTHONPATH=src` or script-local path patching.

Local validation mirrors the core CI commands through `scripts/validate_local.py`
without installing dependencies. CI runs the same validation sequence on
Python 3.11 and 3.12. Fake teacher export artifacts remain generated local state
under the gitignored `artifacts/` directory.
