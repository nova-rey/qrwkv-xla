from __future__ import annotations

import json
import zipfile
from pathlib import Path

from scripts.inventory_research_sources import (
    inventory_source,
    write_inventory_report,
)


def test_inventory_can_read_tiny_fake_zip(tmp_path: Path) -> None:
    source = _fake_zip(tmp_path)

    inventory = inventory_source("fake", source)

    assert inventory.available is True
    assert inventory.source_type == "zip"
    assert inventory.file_count == 4
    assert inventory.top_level_entries == ("fake-repo",)


def test_inventory_lists_readme_config_and_eval_like_files(tmp_path: Path) -> None:
    inventory = inventory_source("fake", _fake_zip(tmp_path))

    assert "fake-repo/README.md" in inventory.readme_like_files
    assert "fake-repo/configs/model.yaml" in inventory.config_like_files
    assert "fake-repo/eval/harness.py" in inventory.eval_like_files


def test_inventory_does_not_extract_or_vendor_code(tmp_path: Path) -> None:
    _ = inventory_source("fake", _fake_zip(tmp_path))

    assert not (tmp_path / "fake-repo").exists()


def test_inventory_json_is_serializable(tmp_path: Path) -> None:
    inventory = inventory_source("fake", _fake_zip(tmp_path))
    output = write_inventory_report((inventory,), tmp_path / "inventory.json")

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["phase"] == "P113"
    assert payload["sources"][0]["name"] == "fake"


def test_missing_source_file_is_unavailable_not_fatal(tmp_path: Path) -> None:
    inventory = inventory_source("missing", tmp_path / "missing.zip")

    assert inventory.available is False
    assert inventory.file_count == 0
    assert inventory.notes == ("source file is unavailable",)


def _fake_zip(tmp_path: Path) -> Path:
    path = tmp_path / "fake.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("fake-repo/README.md", "# fake\n")
        archive.writestr("fake-repo/configs/model.yaml", "model: tiny\n")
        archive.writestr("fake-repo/eval/harness.py", "print('eval')\n")
        archive.writestr("fake-repo/src/module.py", "VALUE = 1\n")
    return path
