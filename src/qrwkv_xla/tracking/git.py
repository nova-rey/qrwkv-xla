from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any


def get_git_metadata(repo_dir: str | Path = ".") -> dict[str, Any]:
    repo_path = Path(repo_dir)
    metadata: dict[str, Any] = {"available": False}
    try:
        commit = _git(repo_path, "rev-parse", "HEAD")
        branch = _git(repo_path, "rev-parse", "--abbrev-ref", "HEAD")
        status_short = _git(repo_path, "status", "--short")
        remote_url = _git_optional(repo_path, "remote", "get-url", "origin")
    except (OSError, subprocess.SubprocessError) as exc:
        metadata["error"] = str(exc)
        return metadata

    metadata.update(
        {
            "available": True,
            "commit": commit,
            "branch": branch,
            "is_dirty": bool(status_short.strip()),
            "status_short": status_short,
            "remote_url": remote_url,
        }
    )
    return metadata


def get_environment_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "jax_available": False,
        "jax_version": None,
        "jax_backend": None,
        "jax_devices": [],
    }
    try:
        from qrwkv_xla.xla.inspect import get_jax_runtime_info

        runtime = get_jax_runtime_info()
        metadata.update(
            {
                "jax_available": runtime.jax_available,
                "jax_version": runtime.jax_version,
                "jax_backend": runtime.default_backend,
                "jax_devices": list(runtime.devices),
                "jax_platforms": list(runtime.platforms),
                "has_cpu": runtime.has_cpu,
                "has_gpu": runtime.has_gpu,
                "has_tpu": runtime.has_tpu,
            }
        )
    except Exception as exc:  # pragma: no cover
        metadata["jax_error"] = str(exc)
    return metadata


def _git(repo_path: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=repo_path,
        capture_output=True,
        check=True,
        text=True,
        timeout=5,
    )
    return completed.stdout.strip()


def _git_optional(repo_path: Path, *args: str) -> str | None:
    try:
        value = _git(repo_path, *args)
    except (OSError, subprocess.SubprocessError):
        return None
    return value or None
