#!/usr/bin/env bash
set -euo pipefail

run_if_available() {
 local cmd="$1"
 shift
 local exe="$cmd"

 if [[ -x ".venv/bin/$cmd" ]]; then
  exe=".venv/bin/$cmd"
 elif ! command -v "$cmd" >/dev/null 2>&1; then
  echo
  echo "== Skipping: $cmd not available =="
  return 0
 fi

 echo
 echo "== Running: $exe $* =="
 "$exe" "$@"
}

if [[ -f pyproject.toml || -d tests ]]; then
 run_if_available ruff check .
 run_if_available black --check .
 run_if_available pytest -q
else
 echo "No obvious Python project markers found; skipping Python validation."
fi

echo
echo "Validation script completed."
