#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$HOME/qrwkv_artifacts}"
TEXTBOOK_URL="${TEXTBOOK_URL:-https://github.com/nova-rey/qrwkv-xla/releases/download/p117-textbook-smoke-v0/p117_teacher_textbook_tiny_gpt2_smoke.tar.gz}"
TEXTBOOK_SHA256="${TEXTBOOK_SHA256:-cbe355a415606012eae4fa856aee180b1b6dc83ee06b4fb70188d57f253d7f23}"
TEXTBOOK_DIR="${TEXTBOOK_DIR:-$ARTIFACT_ROOT/p117_teacher_textbook_tiny_gpt2_smoke}"
TEXTBOOK_TARBALL="${TEXTBOOK_TARBALL:-$ARTIFACT_ROOT/p117_teacher_textbook_tiny_gpt2_smoke.tar.gz}"
PREFLIGHT_DIR="${PREFLIGHT_DIR:-$ARTIFACT_ROOT/p117_tpu_preflight}"
REPORT_TARBALL="${REPORT_TARBALL:-$ARTIFACT_ROOT/p117_tpu_preflight_reports.tar.gz}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
FORCE_EXTRACT="${FORCE_EXTRACT:-0}"

log() {
  printf '[p117-preflight] %s\n' "$*"
}

have() {
  command -v "$1" >/dev/null 2>&1
}

activate_venv_if_present() {
  if [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.venv/bin/activate"
    log "activated venv: $ROOT_DIR/.venv"
  else
    log "warning: .venv not found; using current Python: $(command -v python || true)"
  fi
}

download_textbook() {
  mkdir -p "$ARTIFACT_ROOT" "$PREFLIGHT_DIR"
  if [[ "$FORCE_DOWNLOAD" == "1" || ! -f "$TEXTBOOK_TARBALL" ]]; then
    log "downloading TeacherTextbook: $TEXTBOOK_URL"
    if have curl; then
      curl -L --fail --show-error --output "$TEXTBOOK_TARBALL" "$TEXTBOOK_URL"
    elif have wget; then
      wget -O "$TEXTBOOK_TARBALL" "$TEXTBOOK_URL"
    else
      log "error: curl or wget is required to download textbook"
      exit 1
    fi
  else
    log "reusing existing tarball: $TEXTBOOK_TARBALL"
  fi
}

verify_sha256() {
  log "verifying TeacherTextbook SHA256"
  local actual
  if have sha256sum; then
    actual="$(sha256sum "$TEXTBOOK_TARBALL" | awk '{print $1}')"
  elif have shasum; then
    actual="$(shasum -a 256 "$TEXTBOOK_TARBALL" | awk '{print $1}')"
  else
    log "error: sha256sum or shasum is required"
    exit 1
  fi
  if [[ "$actual" != "$TEXTBOOK_SHA256" ]]; then
    log "error: checksum mismatch"
    log "expected: $TEXTBOOK_SHA256"
    log "actual:   $actual"
    exit 1
  fi
  log "TeacherTextbook SHA256 verified: $actual"
}

extract_textbook() {
  if [[ "$FORCE_EXTRACT" == "1" || ! -d "$TEXTBOOK_DIR" ]]; then
    log "extracting TeacherTextbook to $ARTIFACT_ROOT"
    mkdir -p "$ARTIFACT_ROOT"
    tar -xzf "$TEXTBOOK_TARBALL" -C "$ARTIFACT_ROOT"
  else
    log "reusing extracted textbook: $TEXTBOOK_DIR"
  fi
  if [[ ! -d "$TEXTBOOK_DIR" ]]; then
    log "error: expected textbook directory not found after extract: $TEXTBOOK_DIR"
    exit 1
  fi
}

validate_textbook() {
  log "validating TeacherTextbook"
  python "$ROOT_DIR/scripts/validate_teacher_textbook.py" --path "$TEXTBOOK_DIR"
}

run_readiness() {
  log "running big burn readiness report"
  python "$ROOT_DIR/scripts/run_big_burn_readiness_report.py" \
    --output "$PREFLIGHT_DIR/readiness_report.json"
}

run_dry_run() {
  log "running first serious burn harness in dry_run mode"
  python "$ROOT_DIR/scripts/run_first_serious_burn.py" \
    --output "$PREFLIGHT_DIR/p117_dry_run" \
    --mode dry_run
}

package_reports() {
  log "packaging preflight reports"
  tar -czf "$REPORT_TARBALL" -C "$ARTIFACT_ROOT" "$(basename "$PREFLIGHT_DIR")"
  if have sha256sum; then
    sha256sum "$REPORT_TARBALL" | tee "$REPORT_TARBALL.sha256"
  elif have shasum; then
    shasum -a 256 "$REPORT_TARBALL" | tee "$REPORT_TARBALL.sha256"
  fi
}

print_outputs() {
  log "key outputs:"
  find "$PREFLIGHT_DIR" -maxdepth 3 -type f | sort || true
  log "report bundle: $REPORT_TARBALL"
}

print_manual_real_burn_command() {
  log "manual real-burn command is printed for review only; it is not executed."
  if python "$ROOT_DIR/scripts/run_first_serious_burn.py" --help | grep -q -- '--teacher-textbook'; then
    cat <<EOF
python scripts/run_first_serious_burn.py \\
  --output "$ARTIFACT_ROOT/p117_real_burn" \\
  --mode real \\
  --confirm-serious-burn \\
  --teacher-textbook "$TEXTBOOK_DIR"
EOF
  else
    cat <<EOF
WARNING: scripts/run_first_serious_burn.py does not currently accept --teacher-textbook.
P117 must resolve that handoff before the real run.
Reviewed manual command shape:
python scripts/run_first_serious_burn.py \\
  --output "$ARTIFACT_ROOT/p117_real_burn" \\
  --mode real \\
  --confirm-serious-burn \\
  --teacher-textbook "$TEXTBOOK_DIR"
EOF
  fi
}

main() {
  cd "$ROOT_DIR"
  activate_venv_if_present
  download_textbook
  verify_sha256
  extract_textbook
  validate_textbook
  run_readiness
  run_dry_run
  package_reports
  print_outputs
  print_manual_real_burn_command
  log "preflight complete; no real burn was started"
}

main "$@"
