from __future__ import annotations

from pathlib import Path


def target_bundle_dir(root: str | Path, bundle_id: str) -> Path:
    if not bundle_id.strip():
        raise ValueError("bundle_id must be non-empty")
    return Path(root) / bundle_id


def manifest_path(bundle_dir: str | Path) -> Path:
    return Path(bundle_dir) / "manifest.json"


def shards_dir(bundle_dir: str | Path) -> Path:
    return Path(bundle_dir) / "shards"


def shard_path(bundle_dir: str | Path, index: int) -> Path:
    if index < 0:
        raise ValueError("shard index must be >= 0")
    return shards_dir(bundle_dir) / f"shard_{index:06d}.npz"


def ensure_bundle_layout(bundle_dir: str | Path) -> None:
    bundle_path = Path(bundle_dir)
    bundle_path.mkdir(parents=True, exist_ok=True)
    shards_dir(bundle_path).mkdir(parents=True, exist_ok=True)


def list_shard_paths(bundle_dir: str | Path) -> list[Path]:
    return sorted(shards_dir(bundle_dir).glob("shard_*.npz"))
