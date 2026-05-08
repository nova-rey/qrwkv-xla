# P42 lm_eval-Style Exported-Student Smoke

P42 adds a tiny local evaluation smoke for a P41 HF/safetensors exported
QRWKV-XLA student. The default path is intentionally dependency-light and
offline: it loads `artifacts/p41_hf_safetensors_export_smoke`, scores token-id
continuations from `tests/fixtures/eval/p42_toy_continuations.jsonl`, and writes
results under `artifacts/eval/p42_lm_eval_smoke`.

Run it with:

```bash
python scripts/run_lm_eval_smoke.py \
  --export-dir artifacts/p41_hf_safetensors_export_smoke \
  --out artifacts/eval/p42_lm_eval_smoke \
  --overwrite
```

If the export directory does not exist, the script generates it with the P41
export smoke. If the directory exists but is missing `config.json`,
`model.safetensors`, `qrwkv_xla_export.json`, or `weight_map.json`, the script
fails with a clear missing-file error.

Outputs:

- `results.json`
- `P42_RESULTS.md`
- `p42_results_bundle.tar.gz`

The metrics are finite toy-smoke metrics only: example count, scored token
count, continuation loglikelihood, mean negative loglikelihood, perplexity, and
greedy continuation accuracy.

Official `lm_eval` execution is deferred in P42. The project exposes an
optional `eval` extra as the dependency path for manual future experiments:

```bash
python -m pip install -e ".[eval]"
```

The default smoke does not import or run official `lm_eval`, does not require
network access, and does not make benchmark or model-quality claims.
