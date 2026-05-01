from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from qrwkv_xla.prompting import read_prompt_corpus

ROOT = Path(__file__).resolve().parents[1]


def test_inspect_prompt_corpus_cli() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_prompt_corpus.py",
            "corpora/smoke_prompts.jsonl",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "corpus_id: smoke_prompts" in result.stdout
    assert "prompt_count: 8" in result.stdout


def test_inspect_prompt_corpus_json_cli() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_prompt_corpus.py",
            "corpora/smoke_prompts.jsonl",
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["corpus_id"] == "smoke_prompts"
    assert payload["prompt_count"] == 8


def test_create_manifest_and_split_cli(tmp_path: Path) -> None:
    manifest_path = tmp_path / "smoke.manifest.json"
    split_path = tmp_path / "split.jsonl"

    create = subprocess.run(
        [
            sys.executable,
            "scripts/create_prompt_manifest.py",
            "corpora/smoke_prompts.jsonl",
            "--out",
            str(manifest_path),
            "--description",
            "Tiny checked-in smoke prompt corpus.",
            "--overwrite",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    split = subprocess.run(
        [
            sys.executable,
            "scripts/split_prompt_corpus.py",
            "corpora/smoke_prompts.jsonl",
            "--out",
            str(split_path),
            "--seed",
            "123",
            "--overwrite",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert create.returncode == 0, create.stderr
    assert split.returncode == 0, split.stderr
    assert manifest_path.exists()
    assert split_path.exists()
    assert read_prompt_corpus(split_path).records
