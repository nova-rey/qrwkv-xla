from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from qrwkv_xla.artifacts import (
    TeacherTextbookBuildConfig,
    build_teacher_textbook,
    load_text_examples,
    validate_teacher_textbook,
)
from qrwkv_xla.artifacts._json import read_json_object

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_teacher_textbook.py"


def test_fake_builder_creates_valid_teacher_textbook(tmp_path: Path) -> None:
    output = tmp_path / "teacher_textbook"

    report = build_teacher_textbook(_config(output))

    assert report.status == "pass"
    assert (output / "metadata.json").is_file()
    assert (output / "vocab_contract.json").is_file()
    assert (output / "teacher_manifest.json").is_file()
    assert (output / "emission_config.json").is_file()
    assert (output / "validation_report.json").is_file()
    assert (output / "shards" / "shard-00000.npz").is_file()
    assert validate_teacher_textbook(output).status == "pass"


def test_jsonl_dataset_loader_preserves_order_and_max_examples(tmp_path: Path) -> None:
    dataset = tmp_path / "input.jsonl"
    _write_jsonl(dataset, ["alpha", "beta", "gamma"])

    examples = load_text_examples(dataset, max_examples=2)

    assert [example.example_id for example in examples] == ["ex-0000", "ex-0001"]
    assert [example.text for example in examples] == ["alpha", "beta"]


def test_jsonl_dataset_loader_rejects_empty_text(tmp_path: Path) -> None:
    dataset = tmp_path / "input.jsonl"
    dataset.write_text('{"example_id": "bad", "text": "  "}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="text must be non-empty"):
        load_text_examples(dataset, max_examples=4)


def test_builtin_examples_work_when_dataset_omitted(tmp_path: Path) -> None:
    output = tmp_path / "teacher_textbook"

    report = build_teacher_textbook(_config(output, dataset_path=None))

    assert report.status == "pass"
    metadata = read_json_object(output / "metadata.json")
    assert metadata["num_examples"] == 4
    assert metadata["source"]["kind"] == "builtin_examples"


def test_fake_builder_shard_shapes_and_manifest_match_metadata(tmp_path: Path) -> None:
    dataset = tmp_path / "input.jsonl"
    _write_jsonl(dataset, ["alpha", "beta", "gamma"])
    output = tmp_path / "teacher_textbook"

    build_teacher_textbook(
        _config(output, dataset_path=dataset, max_examples=3, batch_size=2)
    )

    metadata = read_json_object(output / "metadata.json")
    manifest = read_json_object(output / "teacher_manifest.json")
    assert manifest["num_examples"] == metadata["num_examples"] == 3
    assert manifest["sequence_length"] == metadata["sequence_length"] == 8
    assert manifest["vocab_size"] == metadata["vocab_size"] == 17
    assert manifest["shard_count"] == metadata["shard_count"] == 2
    assert manifest["dtype"] == metadata["dtype"] == "float32"

    with np.load(output / "shards" / "shard-00000.npz", allow_pickle=False) as shard:
        assert shard["input_ids"].shape == (2, 8)
        assert shard["attention_mask"].shape == (2, 8)
        assert shard["logits"].shape == (2, 8, 17)
    with np.load(output / "shards" / "shard-00001.npz", allow_pickle=False) as shard:
        assert shard["input_ids"].shape == (1, 8)
        assert shard["attention_mask"].shape == (1, 8)
        assert shard["logits"].shape == (1, 8, 17)


def test_emission_config_records_fake_mode_and_no_sampling(tmp_path: Path) -> None:
    output = tmp_path / "teacher_textbook"

    build_teacher_textbook(_config(output))

    emission = read_json_object(output / "emission_config.json")
    report = read_json_object(output / "validation_report.json")
    assert emission["teacher_mode"] == "fake"
    assert emission["sampling_used"] is False
    assert report["status"] == "pass"


def test_cli_writes_valid_artifact_without_hf_internet_gpu_or_tpu(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "input.jsonl"
    _write_jsonl(dataset, ["alpha", "beta"])
    output = tmp_path / "teacher_textbook"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--teacher-mode",
            "fake",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--sequence-length",
            "8",
            "--batch-size",
            "2",
            "--max-examples",
            "2",
            "--logits-dtype",
            "float32",
            "--vocab-size",
            "17",
        ],
        check=False,
        env={"PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "status=pass" in result.stdout
    assert validate_teacher_textbook(output).status == "pass"


def test_hf_mode_fails_clearly_without_implicit_downloads(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="teacher-mode=hf is intentionally guarded"):
        build_teacher_textbook(
            TeacherTextbookBuildConfig(
                output_dir=tmp_path / "teacher_textbook",
                teacher_mode="hf",
                teacher_model_id="sshleifer/tiny-gpt2",
                allow_downloads=False,
                local_files_only=True,
            )
        )


def test_builder_does_not_train_or_start_real_burn(tmp_path: Path) -> None:
    output = tmp_path / "teacher_textbook"

    build_teacher_textbook(_config(output))

    claims = read_json_object(output / "teacher_manifest.json")["claims_not_made"]
    assert "no_training_claim" in claims
    assert "no_remote_teacher_service_claim" in claims


def _config(
    output: Path,
    *,
    dataset_path: Path | None = None,
    max_examples: int = 4,
    batch_size: int = 2,
) -> TeacherTextbookBuildConfig:
    return TeacherTextbookBuildConfig(
        output_dir=output,
        dataset_path=dataset_path,
        teacher_mode="fake",
        sequence_length=8,
        batch_size=batch_size,
        max_examples=max_examples,
        logits_dtype="float32",
        local_files_only=True,
        allow_downloads=False,
        seed=123,
        overwrite=False,
        vocab_size=17,
    )


def _write_jsonl(path: Path, texts: list[str]) -> None:
    rows = [
        json.dumps({"example_id": f"ex-{idx:04d}", "text": text})
        for idx, text in enumerate(texts)
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
