from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any

from qrwkv_xla.eval.exported_student import run_toy_exported_student_eval
from qrwkv_xla.export.hf_safetensors import (
    CONFIG_NAME,
    EXPORT_METADATA_NAME,
    MODEL_NAME,
    WEIGHT_MAP_NAME,
)

DEFAULT_EXPORT_DIR = Path("artifacts/p41_hf_safetensors_export_smoke")
DEFAULT_TASK = Path("tests/fixtures/eval/p42_toy_continuations.jsonl")
DEFAULT_OUT = Path("artifacts/eval/p42_lm_eval_smoke")
RESULTS_JSON = "results.json"
RESULTS_MD = "P42_RESULTS.md"
BUNDLE = "p42_results_bundle.tar.gz"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run P42 lm_eval-style toy smoke on a P41 exported student"
    )
    parser.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR))
    parser.add_argument("--task", default=str(DEFAULT_TASK))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_lm_eval_smoke(
        export_dir=Path(args.export_dir),
        task_path=Path(args.task),
        out_dir=Path(args.out),
        overwrite=args.overwrite,
    )
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"p42_results: {report['files']['results_json']}")
        print(f"passed: {report['passed']}")


def run_lm_eval_smoke(
    *,
    export_dir: Path,
    task_path: Path,
    out_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    export_created = ensure_p41_export(export_dir)
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"eval output already exists at {out_dir}; pass --overwrite"
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_toy_exported_student_eval(export_dir=export_dir, task_path=task_path)
    report = result.to_json()
    report.update(
        {
            "passed": (
                result.num_examples > 0
                and result.num_tokens_scored > 0
                and result.perplexity > 0.0
            ),
            "export_created": export_created,
            "files": {
                "results_json": str(out_dir / RESULTS_JSON),
                "results_markdown": str(out_dir / RESULTS_MD),
                "bundle": str(out_dir / BUNDLE),
            },
        }
    )
    (out_dir / RESULTS_JSON).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / RESULTS_MD).write_text(_markdown_report(report), encoding="utf-8")
    _write_bundle(out_dir)
    if not report["passed"]:
        raise ValueError("P42 lm_eval-style toy smoke did not pass")
    return report


def ensure_p41_export(export_dir: Path) -> bool:
    required = (CONFIG_NAME, MODEL_NAME, EXPORT_METADATA_NAME, WEIGHT_MAP_NAME)
    if export_dir.exists():
        missing = [name for name in required if not (export_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "P41 export directory is missing required files: "
                + ", ".join(str(export_dir / name) for name in missing)
            )
        return False

    _load_p41_smoke_runner()(export_dir, overwrite=True)
    return True


def _load_p41_smoke_runner() -> Any:
    script_path = Path(__file__).resolve().parent / "run_export_smoke.py"
    spec = importlib.util.spec_from_file_location("p41_run_export_smoke", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load P41 export smoke script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_export_smoke


def _write_bundle(out_dir: Path) -> Path:
    bundle_path = out_dir / BUNDLE
    with tarfile.open(bundle_path, "w:gz") as tar:
        for name in (RESULTS_JSON, RESULTS_MD):
            tar.add(out_dir / name, arcname=name)
    return bundle_path


def _markdown_report(report: dict[str, Any]) -> str:
    status = "PASS" if report["passed"] else "FAIL"
    return (
        "# P42 lm_eval-Style Toy Exported-Student Smoke\n\n"
        f"- status: {status}\n"
        f"- mode: `{report['mode']}`\n"
        f"- export dir: `{report['export_dir']}`\n"
        f"- task path: `{report['task_path']}`\n"
        f"- examples: {report['num_examples']}\n"
        f"- tokens scored: {report['num_tokens_scored']}\n"
        f"- mean negative loglikelihood: {report['mean_neg_loglikelihood']}\n"
        f"- perplexity: {report['perplexity']}\n"
        f"- greedy continuation accuracy: {report['greedy_accuracy']}\n\n"
        "This is an lm_eval-style local toy harness over token-id fixtures. "
        "Official `lm_eval` execution is explicitly deferred for P42 so default "
        "tests stay offline, tiny, and dependency-light.\n"
    )


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        print(f"P42 lm_eval smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except (FileExistsError, FileNotFoundError, ValueError, TypeError) as exc:
        print(f"P42 lm_eval smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
