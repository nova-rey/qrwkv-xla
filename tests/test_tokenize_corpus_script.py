from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_tokenize_corpus_script_smoke(tmp_path: Path) -> None:
    corpus_path = tmp_path / "prompts.jsonl"
    corpus_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "one", "text": "alpha", "split": "train"}),
                json.dumps({"id": "two", "text": "beta", "split": "train"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "tok"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/tokenize_corpus.py",
            "--input",
            str(corpus_path),
            "--out",
            str(output_dir),
            "--tokenizer-backend",
            "smoke",
            "--sequence-length",
            "3",
            "--shard-size-tokens",
            "6",
            "--overwrite",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Wrote tokenized corpus:" in completed.stdout
    assert "sequences: 3" in completed.stdout
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "shards" / "shard-00000.npz").is_file()
