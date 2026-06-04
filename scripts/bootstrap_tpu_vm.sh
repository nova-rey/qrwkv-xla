#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

log() {
  printf '[bootstrap] %s\n' "$*"
}

have() {
  command -v "$1" >/dev/null 2>&1
}

print_host_info() {
  log "repo: $ROOT_DIR"
  log "uname: $(uname -a)"
  if have lsb_release; then
    lsb_release -a 2>/dev/null || true
  elif [[ -r /etc/os-release ]]; then
    cat /etc/os-release
  fi
}

missing_commands() {
  local missing=()
  for cmd in git python3 curl tar; do
    if ! have "$cmd"; then
      missing+=("$cmd")
    fi
  done
  printf '%s\n' "${missing[@]}"
}

install_system_packages_if_possible() {
  mapfile -t missing < <(missing_commands)
  if [[ "${#missing[@]}" -eq 0 ]]; then
    log "required baseline commands already present"
    return
  fi
  log "missing baseline commands: ${missing[*]}"
  if ! have apt-get; then
    log "apt-get not available; install missing commands manually"
    return
  fi
  if have sudo; then
    log "installing minimal packages with sudo apt-get"
    sudo apt-get update
    sudo apt-get install -y \
      git git-lfs python3 python3-venv python3-pip \
      curl wget ca-certificates coreutils
  elif [[ "$(id -u)" == "0" ]]; then
    log "installing minimal packages with apt-get as root"
    apt-get update
    apt-get install -y \
      git git-lfs python3 python3-venv python3-pip \
      curl wget ca-certificates coreutils
  else
    log "sudo unavailable; install packages manually: ${missing[*]}"
  fi
}

create_or_reuse_venv() {
  if [[ ! -d "$VENV_DIR" ]]; then
    log "creating venv: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  else
    log "reusing venv: $VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -e "$ROOT_DIR"
  python -m pip install pytest ruff black PyYAML
}

diagnose_python_runtime() {
  python - <<'PY'
import importlib
import shutil
import subprocess
import sys

print("python_executable:", sys.executable)
print("python_version:", sys.version.replace("\n", " "))
try:
    import pip
    print("pip_version:", pip.__version__)
except Exception as exc:
    print("pip_version: unavailable", repr(exc))

versions = {}
for name in ("numpy", "torch", "jax", "jaxlib"):
    try:
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "unknown")
        print(f"{name}_version:", versions[name])
    except Exception as exc:
        print(f"{name}_version: unavailable {exc!r}")

try:
    import jax
    print("jax_default_backend:", jax.default_backend())
    print("jax_devices:", jax.devices())
except Exception as exc:
    print("jax_devices: unavailable", repr(exc))

numpy_version = versions.get("numpy")
torch_version = versions.get("torch")
if numpy_version and torch_version and torch_version.startswith("2.2."):
    try:
        numpy_major = int(numpy_version.split(".", 1)[0])
    except ValueError:
        numpy_major = 0
    if numpy_major >= 2:
        print("WARNING: torch 2.2.x with NumPy 2.x may be incompatible.")
        print('Suggested venv-local fix: python -m pip install --force-reinstall "numpy<2"')

for cmd in ("git", "curl", "wget", "tar"):
    path = shutil.which(cmd)
    print(f"{cmd}_path:", path or "missing")
    if path:
        try:
            out = subprocess.run([cmd, "--version"], text=True, capture_output=True, timeout=5)
            print(f"{cmd}_version:", (out.stdout or out.stderr).splitlines()[0])
        except Exception as exc:
            print(f"{cmd}_version: unavailable {exc!r}")
PY
}

main() {
  cd "$ROOT_DIR"
  print_host_info
  install_system_packages_if_possible
  create_or_reuse_venv
  diagnose_python_runtime
  log "bootstrap complete"
  log "next command: bash scripts/run_p117_preflight.sh"
}

main "$@"
