# P88 TPU Compile / Performance Smoke

P88 adds a tiny opt-in Pallas TPU smoke harness. It inspects the JAX runtime,
detects TPU devices, imports the Pallas WKV path, and only attempts lowering,
compile, execution, and a tiny numeric check when TPU devices are present.

What P88 proves on TPU: the current tiny opt-in Pallas WKV path can import,
lower/JIT, execute, and match a tiny reference check in that environment.

What P88 does not prove: production Pallas readiness, training readiness,
throughput, model quality, full-model parity, or Pallas default readiness.

## Local CPU Preflight

```bash
git checkout main
git pull
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
python scripts/run_pallas_tpu_smoke.py \
  --output artifacts/p88_tpu_compile_smoke/pallas_tpu_compile_smoke.json
cat artifacts/p88_tpu_compile_smoke/pallas_tpu_compile_smoke.json
```

Expected without TPU:

```text
status: unavailable
reason: no_tpu_devices_detected
```

## Google Colab / Kaggle TPU Notebook

Notebook cell 1:

```bash
!git clone https://github.com/nova-rey/qrwkv-xla.git
%cd qrwkv-xla
!python -m pip install -U pip
!python -m pip install -e ".[dev]"
```

Notebook cell 2:

```python
import jax
print("jax", jax.__version__)
print("backend", jax.default_backend())
print("devices", jax.devices())
```

Notebook cell 3:

```bash
!python scripts/run_pallas_tpu_smoke.py \
  --output artifacts/p88_tpu_compile_smoke/pallas_tpu_compile_smoke.json
!cat artifacts/p88_tpu_compile_smoke/pallas_tpu_compile_smoke.json
```

## Cloud TPU VM

```bash
git clone https://github.com/nova-rey/qrwkv-xla.git
cd qrwkv-xla
python -m pip install -U pip
python -m pip install "jax[tpu]" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -m pip install -e ".[dev]"
python scripts/run_pallas_tpu_smoke.py \
  --output artifacts/p88_tpu_compile_smoke/pallas_tpu_compile_smoke.json
cat artifacts/p88_tpu_compile_smoke/pallas_tpu_compile_smoke.json
```

Outcomes:

- CPU/no TPU: `status=unavailable`.
- TPU smoke pass: `status=pass`.
- Compile/runtime failure: `status=fail`.

Notebook and TPU VM JAX setup varies by provider; the script reports the
observed devices and backend instead of assuming a specific TPU configuration.
