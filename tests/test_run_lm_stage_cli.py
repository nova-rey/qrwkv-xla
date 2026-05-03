from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from qrwkv_xla.generation import SmokeTokenizer
from qrwkv_xla.lm.tokenized_corpus import write_tokenized_corpus_from_prompt_jsonl

ROOT = Path(__file__).resolve().parents[1]


def test_run_lm_stage_cli_smoke(tmp_path: Path) -> None:
    checkpoint_out = tmp_path / "checkpoints" / "lm_cli"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_lm_stage.py",
            "--config",
            "configs/lm_stage3_smoke.yaml",
            "--max-steps",
            "1",
            "--checkpoint-out",
            str(checkpoint_out),
            "--checkpoint-overwrite",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "mode: lm_stage3_ce" in completed.stdout
    assert "final_ce_loss:" in completed.stdout
    assert (checkpoint_out / "checkpoint.json").is_file()


def test_run_lm_stage_cli_tokenized_smoke(tmp_path: Path) -> None:
    corpus_path = tmp_path / "prompts.jsonl"
    corpus_path.write_text(
        '{"id":"one","text":"alpha","split":"train"}\n'
        '{"id":"two","text":"beta","split":"train"}\n',
        encoding="utf-8",
    )
    tokenized_dir = tmp_path / "tokenized"
    checkpoint_out = tmp_path / "checkpoints" / "lm_cli_tokenized"
    write_tokenized_corpus_from_prompt_jsonl(
        corpus_path,
        tokenized_dir,
        tokenizer=SmokeTokenizer(),
        sequence_length=3,
        shard_size_tokens=6,
        overwrite=True,
        created_at="2026-05-03T22:20:00+00:00",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_lm_stage.py",
            "--config",
            "configs/lm_stage3_tokenized_smoke.yaml",
            "--tokenized-corpus",
            str(tokenized_dir),
            "--sequence-length",
            "3",
            "--batch-size",
            "2",
            "--max-steps",
            "1",
            "--checkpoint-out",
            str(checkpoint_out),
            "--checkpoint-overwrite",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"tokenized_corpus: {tokenized_dir}" in completed.stdout
    assert "final_ce_loss:" in completed.stdout
    assert (checkpoint_out / "checkpoint.json").is_file()
