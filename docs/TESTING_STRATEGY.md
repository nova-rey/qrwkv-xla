# Testing Strategy

## Tier 0: import/layout tests
Runs anywhere.

## Tier 1: CPU config/smoke tests
Runs on the local VM with no GPU.

## Tier 2: JAX CPU train-step tests
Later phase.

## Tier 3: TPU smoke tests
Kaggle/Colab/free TPU when available. Should be tiny and fast.

## Tier 4: large TPU scale tests
Only after grant or paid TPU access.

## Phase 0 scope

Phase 0 only requires real Tier 0 tests:
- `tests/test_imports.py`
- `tests/test_project_layout.py`

Those tests should pass even if JAX or TPU libraries are unavailable.
