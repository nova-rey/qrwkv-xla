# P117 TPU Runbook

## Purpose

This runbook gives a human operator a safe, repeatable path from a fresh TPU VM
to P117 preflight evidence. It uses the P116.3 bootstrap and preflight scripts
to set up the repo environment, download the tiny-HF TeacherTextbook artifact,
verify it, validate it, run readiness, and run the first serious burn harness in
`dry_run` mode.

## What This Proves

- the TPU VM can run the QRWKV-XLA repo tooling
- the released tiny-HF TeacherTextbook can be downloaded and checksum-verified
- the extracted TeacherTextbook validates
- readiness reporting runs on the VM
- the guarded burn harness can run in `dry_run` mode
- preflight reports can be packaged for review

## What This Does Not Prove

- the real burn succeeded
- useful model quality
- large-corpus training readiness
- Qwen parity
- KVM, FLA, or Vocab C readiness
- production Pallas readiness
- full HF-native student integration

The released textbook proves handoff and ingestion. It is not large enough to
train a useful model.

## Prerequisites

- Google Cloud CLI authenticated on the operator machine
- a TPU VM or queued resource available
- repo clone on the TPU VM
- network access from the TPU VM to GitHub Releases
- enough local disk for the mini textbook and preflight reports

## TPU VM Creation Notes

- `v6e-16` is the intended 4x4 Gen 6 / Trillium slice.
- `v2-alpha-tpuv6e` is the intended runtime used during setup.
- Some zones may return `Reservation not found`.
- Some zones may return `Insufficient capacity`.
- Queued resources can be used when capacity is unavailable.

Example queued resource:

```bash
gcloud compute tpus queued-resources create qrwkv-p117-v6e-16-qr \
  --zone europe-west4-a \
  --node-id qrwkv-p117-v6e-16 \
  --accelerator-type v6e-16 \
  --runtime-version v2-alpha-tpuv6e \
  --valid-until-duration=4h
```

SSH:

```bash
gcloud compute tpus tpu-vm ssh qrwkv-p117-v6e-16 \
  --zone europe-west4-a
```

## Bootstrap Fresh TPU VM

Fresh clone:

```bash
git clone https://github.com/nova-rey/qrwkv-xla.git
cd qrwkv-xla
bash scripts/bootstrap_tpu_vm.sh
```

Existing clone:

```bash
cd ~/qrwkv-xla
git pull --ff-only
bash scripts/bootstrap_tpu_vm.sh
```

The bootstrap script installs minimal packages when possible, creates/reuses
`.venv`, installs the repo and baseline test tools, prints Python dependency
diagnostics, warns about the NumPy 2 + old torch issue when detected, and prints
the next preflight command.

## Run P117 Preflight

```bash
bash scripts/run_p117_preflight.sh
```

Default artifact:

```text
https://github.com/nova-rey/qrwkv-xla/releases/download/p117-textbook-smoke-v0/p117_teacher_textbook_tiny_gpt2_smoke.tar.gz
```

Expected SHA256:

```text
cbe355a415606012eae4fa856aee180b1b6dc83ee06b4fb70188d57f253d7f23
```

Useful overrides:

```bash
TEXTBOOK_URL=... \
TEXTBOOK_SHA256=... \
ARTIFACT_ROOT=$HOME/qrwkv_artifacts \
PREFLIGHT_DIR=$HOME/qrwkv_artifacts/p117_tpu_preflight \
bash scripts/run_p117_preflight.sh
```

## Verify TeacherTextbook

The preflight script runs:

```bash
python scripts/validate_teacher_textbook.py \
  --path "$HOME/qrwkv_artifacts/p117_teacher_textbook_tiny_gpt2_smoke"
```

Expected validation status is `pass`.

## Inspect Reports

Default report directory:

```text
$HOME/qrwkv_artifacts/p117_tpu_preflight
```

Packaged report bundle:

```text
$HOME/qrwkv_artifacts/p117_tpu_preflight_reports.tar.gz
```

The preflight script writes a SHA256 file next to the report bundle when
`sha256sum` or `shasum` is available.

## Manual Real-Burn Command

The preflight script prints a reviewed real-burn command but does not execute
it. If the current burn CLI does not support `--teacher-textbook`, the script
prints that warning explicitly.

Reviewed command shape:

```bash
python scripts/run_first_serious_burn.py \
  --output "$HOME/qrwkv_artifacts/p117_real_burn" \
  --mode real \
  --confirm-serious-burn \
  --teacher-textbook "$HOME/qrwkv_artifacts/p117_teacher_textbook_tiny_gpt2_smoke"
```

Do not run real mode until preflight reports are reviewed and the P117 handoff
from TeacherTextbook to burn config is resolved.

## Stop / Start / Delete TPU VM

```bash
gcloud compute tpus tpu-vm stop qrwkv-p117-v6e-16 --zone europe-west4-a
gcloud compute tpus tpu-vm start qrwkv-p117-v6e-16 --zone europe-west4-a
gcloud compute tpus tpu-vm delete qrwkv-p117-v6e-16 --zone europe-west4-a
```

## Troubleshooting

Missing `git`, `python3`, `venv`, `curl`, or `wget`:
run `bash scripts/bootstrap_tpu_vm.sh`; if `apt-get` or `sudo` is unavailable,
install the missing package manually.

Repo install failure:
activate `.venv`, upgrade `pip`, then rerun `python -m pip install -e .`.

TeacherTextbook checksum mismatch:
delete the tarball, rerun with `FORCE_DOWNLOAD=1`, and confirm the expected
SHA256 matches the release asset.

TeacherTextbook validation fail:
inspect `metadata.json`, `teacher_manifest.json`, `vocab_contract.json`, and
the `shards/` directory. Do not proceed to real burn.

Readiness blockers:
open `$PREFLIGHT_DIR/readiness_report.json` and resolve blockers before P117.

Dry-run blockers:
open `$PREFLIGHT_DIR/p117_dry_run/burn_report.json` and resolve blockers before
P117.

Burn CLI missing `--teacher-textbook`:
P116.3 preflight warns about this. P117 must resolve the handoff before a real
run.

NumPy 2 + old torch warning:
if torch `2.2.x` and NumPy `2.x` are both installed in the venv, run:

```bash
python -m pip install --force-reinstall "numpy<2"
```

No TPU devices visible to JAX:
check runtime version, TPU VM health, `jax.devices()`, `TPU_NAME`, `XLA_FLAGS`,
and whether `libtpu` is available in the runtime.

Insufficient capacity / queued resource still waiting:
check queued-resource status, try another supported zone, or wait for capacity.

## Claims Not Made

P116.3 does not run the real burn, train, add a remote teacher service, add
Qwen support, add tokenizer remapping, add a full HF wrapper, implement KVM,
add FLA, implement Vocab C, change model/runtime/math behavior, or promote
Pallas.
