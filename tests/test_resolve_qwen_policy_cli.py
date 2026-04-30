from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "resolve_qwen_policy.py"


def test_resolve_qwen_policy_cli_allows_unresolved() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "Qwen3.latest", "--allow-unresolved"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "label: Qwen3.latest" in result.stdout
    assert "is_resolved: False" in result.stdout


def test_resolve_qwen_policy_cli_json_output() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "Qwen3.latest",
            "--allow-unresolved",
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["label"] == "Qwen3.latest"
    assert payload["is_resolved"] is False
    assert "notes" in payload


def test_resolve_qwen_policy_cli_rejects_unresolved_by_default() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "Qwen3.latest"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "is unresolved" in result.stderr
