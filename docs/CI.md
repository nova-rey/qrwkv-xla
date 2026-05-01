# CI

QRWKV-XLA CI runs on `ubuntu-latest` for Python 3.11 and 3.12. Each job uses
Strategy A: install the project once with an editable install, then run scripts
and tests against the installed package.

## Validation Steps

CI runs these commands in order:

```bash
python -m compileall src scripts tests
python scripts/validate_pipeline.py
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

`scripts/validate_pipeline.py` runs the canonical safe end-to-end validation
path. It is CPU-safe, offline, and uses only `.[dev]` by default. Optional HF
validation (`--include-hf`) and hard TPU validation (`--require-tpu`) are not
part of default CI.
The default path also inspects the smoke prompt corpus, creates its manifest,
dry-runs the Qwen corpus config without importing Hugging Face modules, and
runs a tiny greedy generation smoke from a logits-capable checkpoint, and runs
the non-strict regression evaluation harness on fixed prompts.

## Local Mirror

Create or activate a local virtual environment, then install the package in
editable mode with development tools:

```bash
python -m pip install -e ".[dev]"
python scripts/validate_local.py
```

`scripts/validate_local.py` does not install dependencies. It only mirrors the
CI validation commands, prints each command, and fails fast on the first error.

## Editable Install Policy

Editable install is the blessed workflow because scripts, tests, and CI should
all import `qrwkv_xla` through packaging metadata instead of relying on
`PYTHONPATH=src` or script-local path mutation. This keeps CLI behavior closer
to how the package will run once installed normally, while still allowing source
edits to take effect immediately during development.

## Generated Artifacts

Validation writes fake teacher target bundles under `artifacts/`, checkpoints
under `checkpoints/`, local runs under `runs/`, and generation/evaluation
artifacts under `eval_outputs/`. Those directories are gitignored and should remain
local/generated state rather than committed repository content.
