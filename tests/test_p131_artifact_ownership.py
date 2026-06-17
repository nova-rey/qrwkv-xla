from __future__ import annotations

from pathlib import Path

from qrwkv_xla.burn import (
    canonical_only_path,
    expected_canonical_path,
    expected_per_process_paths,
    is_canonical_process,
    per_process_path,
    write_json_canonical,
    write_json_per_process,
)


def test_process_zero_is_canonical() -> None:
    assert is_canonical_process(0) is True
    assert is_canonical_process(1) is False
    assert is_canonical_process(3) is False


def test_canonical_only_path_noops_for_nonzero_process(tmp_path: Path) -> None:
    assert canonical_only_path(tmp_path, "checkpoint.json", 0) == (
        tmp_path / "checkpoint.json"
    )
    assert canonical_only_path(tmp_path, "checkpoint.json", 2) is None


def test_per_process_paths_include_process_index(tmp_path: Path) -> None:
    assert per_process_path(tmp_path, "burn_report", 2) == (
        tmp_path / "burn_report_process_2.json"
    )
    assert expected_canonical_path(tmp_path, "burn_report.json") == (
        tmp_path / "burn_report.json"
    )
    assert expected_per_process_paths(tmp_path, "burn_report", 4) == (
        str(tmp_path / "burn_report_process_0.json"),
        str(tmp_path / "burn_report_process_1.json"),
        str(tmp_path / "burn_report_process_2.json"),
        str(tmp_path / "burn_report_process_3.json"),
    )


def test_per_process_paths_are_unique(tmp_path: Path) -> None:
    paths = [per_process_path(tmp_path, "sync_report", index) for index in range(8)]

    assert len(paths) == len(set(paths))


def test_write_json_canonical_skips_nonzero_process(tmp_path: Path) -> None:
    path = write_json_canonical(
        {"ok": True}, tmp_path, "canonical.json", process_index=2
    )

    assert path is None
    assert not (tmp_path / "canonical.json").exists()


def test_write_json_canonical_writes_process_zero(tmp_path: Path) -> None:
    path = write_json_canonical(
        {"ok": True}, tmp_path, "canonical.json", process_index=0
    )

    assert path == tmp_path / "canonical.json"
    assert path.is_file()


def test_write_json_per_process_always_uses_process_specific_path(
    tmp_path: Path,
) -> None:
    path = write_json_per_process({"ok": True}, tmp_path, "diagnostic", process_index=3)

    assert path == tmp_path / "diagnostic_process_3.json"
    assert path.is_file()
