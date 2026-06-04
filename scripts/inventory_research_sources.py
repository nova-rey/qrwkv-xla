#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResearchSourceInventory:
    name: str
    path: str
    available: bool
    source_type: str
    file_count: int
    top_level_entries: tuple[str, ...]
    readme_like_files: tuple[str, ...]
    config_like_files: tuple[str, ...]
    eval_like_files: tuple[str, ...]
    notes: tuple[str, ...] = ()

    def to_report(self) -> dict[str, Any]:
        return asdict(self)


def inventory_source(name: str, path: str | Path | None) -> ResearchSourceInventory:
    if path is None:
        return _unavailable(name=name, path="", note="source path was not provided")
    source_path = Path(path)
    if not source_path.is_file():
        return _unavailable(
            name=name,
            path=str(source_path),
            note="source file is unavailable",
        )
    if source_path.suffix.lower() == ".zip":
        return _inventory_zip(name=name, path=source_path)
    if source_path.suffix.lower() == ".pdf":
        return ResearchSourceInventory(
            name=name,
            path=str(source_path),
            available=True,
            source_type="pdf",
            file_count=1,
            top_level_entries=(source_path.name,),
            readme_like_files=(),
            config_like_files=(),
            eval_like_files=(),
            notes=(
                "pdf metadata/file presence recorded; text extraction not attempted",
            ),
        )
    return ResearchSourceInventory(
        name=name,
        path=str(source_path),
        available=True,
        source_type="file",
        file_count=1,
        top_level_entries=(source_path.name,),
        readme_like_files=(),
        config_like_files=(),
        eval_like_files=(),
        notes=("generic file recorded",),
    )


def write_inventory_report(
    inventories: tuple[ResearchSourceInventory, ...],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": "P113",
        "scope": "radlads2_fla_kvm_research_source_inventory",
        "sources": [item.to_report() for item in inventories],
        "claims_not_made": [
            "external_repos_vendored",
            "external_modules_imported",
            "training_started",
            "fla_dependency_added",
            "kvm_implemented",
        ],
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inventory P113 uploaded research source files without extraction."
    )
    parser.add_argument("--distillation-fla", type=Path, default=None)
    parser.add_argument("--hfattnconv", type=Path, default=None)
    parser.add_argument("--kvm-paper", type=Path, default=None)
    parser.add_argument("--vocab-c", type=Path, default=None)
    parser.add_argument("--tier3", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/p113_research_intake/source_inventory.json"),
    )
    args = parser.parse_args()

    inventories = (
        inventory_source("distillation-fla", args.distillation_fla),
        inventory_source("hfattnconv", args.hfattnconv),
        inventory_source("kvm-paper", args.kvm_paper),
        inventory_source("vocab-c", args.vocab_c),
        inventory_source("3tier", args.tier3),
    )
    output = write_inventory_report(inventories, args.output)
    available = sum(item.available for item in inventories)
    print(f"sources={len(inventories)} available={available} report={output}")
    return 0


def _inventory_zip(name: str, path: Path) -> ResearchSourceInventory:
    with zipfile.ZipFile(path) as archive:
        names = tuple(info.filename for info in archive.infolist() if not info.is_dir())
    return ResearchSourceInventory(
        name=name,
        path=str(path),
        available=True,
        source_type="zip",
        file_count=len(names),
        top_level_entries=_top_level_entries(names),
        readme_like_files=_matching(names, ("readme",)),
        config_like_files=_matching(names, ("config", ".yaml", ".json")),
        eval_like_files=_matching(names, ("eval", "harness")),
    )


def _unavailable(name: str, path: str, note: str) -> ResearchSourceInventory:
    return ResearchSourceInventory(
        name=name,
        path=path,
        available=False,
        source_type="unavailable",
        file_count=0,
        top_level_entries=(),
        readme_like_files=(),
        config_like_files=(),
        eval_like_files=(),
        notes=(note,),
    )


def _top_level_entries(names: tuple[str, ...]) -> tuple[str, ...]:
    roots = sorted({name.split("/", 1)[0] for name in names if name})
    return tuple(roots)


def _matching(names: tuple[str, ...], needles: tuple[str, ...]) -> tuple[str, ...]:
    matches = []
    for name in names:
        lowered = name.lower()
        if any(needle in lowered for needle in needles):
            matches.append(name)
    return tuple(matches[:32])


if __name__ == "__main__":
    raise SystemExit(main())
