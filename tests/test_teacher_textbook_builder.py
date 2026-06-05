from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_hf_mode_missing_optional_deps_fails_clearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_import(name: str):
        raise ImportError(name)

    monkeypatch.setattr(
        "qrwkv_xla.artifacts.teacher_textbook_builder.import_module",
        missing_import,
    )

    with pytest.raises(RuntimeError, match="teacher-mode=hf requires optional"):
        build_teacher_textbook(
            TeacherTextbookBuildConfig(
                output_dir=tmp_path / "teacher_textbook",
                teacher_mode="hf",
                teacher_model_id="sshleifer/tiny-gpt2",
                allow_downloads=False,
                local_files_only=True,
            )
        )


def test_hf_mode_builds_valid_artifact_with_mocked_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, list[bool]] = {"tokenizer": [], "model": []}
    _install_mock_hf_modules(monkeypatch, calls=calls)
    dataset = tmp_path / "input.jsonl"
    _write_jsonl(dataset, ["alpha", "beta", "gamma"])
    output = tmp_path / "teacher_textbook"

    report = build_teacher_textbook(
        TeacherTextbookBuildConfig(
            output_dir=output,
            dataset_path=dataset,
            teacher_mode="hf",
            teacher_model_id="local/tiny-hf",
            sequence_length=8,
            batch_size=2,
            max_examples=3,
            logits_dtype="float32",
            local_files_only=True,
            allow_downloads=False,
        )
    )

    assert report.status == "pass"
    assert validate_teacher_textbook(output).status == "pass"
    assert calls["tokenizer"] == [True]
    assert calls["model"] == [True]
    manifest = read_json_object(output / "teacher_manifest.json")
    emission = read_json_object(output / "emission_config.json")
    vocab = read_json_object(output / "vocab_contract.json")
    assert manifest["teacher_backend_type"] == "hf"
    assert manifest["target_type"] == "dense_logits"
    assert manifest["vocab_size"] == 11
    assert emission["teacher_mode"] == "hf"
    assert emission["sampling_used"] is False
    assert emission["local_files_only"] is True
    assert emission["allow_downloads"] is False
    assert vocab["vocab_size"] == 11
    with np.load(output / "shards" / "shard-00000.npz", allow_pickle=False) as shard:
        assert shard["input_ids"].shape == (2, 8)
        assert shard["attention_mask"].shape == (2, 8)
        assert shard["logits"].shape == (2, 8, 11)


def test_hf_mode_allows_downloads_only_when_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, list[bool]] = {"tokenizer": [], "model": []}
    _install_mock_hf_modules(monkeypatch, calls=calls)

    build_teacher_textbook(
        TeacherTextbookBuildConfig(
            output_dir=tmp_path / "teacher_textbook",
            teacher_mode="hf",
            teacher_model_id="local/tiny-hf",
            sequence_length=8,
            batch_size=2,
            max_examples=2,
            local_files_only=False,
            allow_downloads=True,
        )
    )

    assert calls["tokenizer"] == [False]
    assert calls["model"] == [False]


def test_hf_mode_can_emit_topk_tail_without_persisting_dense_logits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, list[bool]] = {"tokenizer": [], "model": []}
    _install_mock_hf_modules(monkeypatch, calls=calls)
    output = tmp_path / "teacher_textbook"

    report = build_teacher_textbook(
        TeacherTextbookBuildConfig(
            output_dir=output,
            teacher_mode="hf",
            teacher_model_id="local/tiny-hf",
            sequence_length=8,
            batch_size=2,
            max_examples=2,
            logits_dtype="float32",
            local_files_only=True,
            allow_downloads=False,
            target_type="topk_with_tail_v0",
            top_k=4,
            top_log_probs_dtype="float32",
        )
    )

    assert report.status == "pass"
    assert validate_teacher_textbook(output).status == "pass"
    assert calls["tokenizer"] == [True]
    assert calls["model"] == [True]
    manifest = read_json_object(output / "teacher_manifest.json")
    assert manifest["target_type"] == "topk_with_tail_v0"
    assert manifest["top_k"] == 4
    with np.load(output / "shards" / "shard-00000.npz", allow_pickle=False) as shard:
        assert "logits" not in shard.files
        assert shard["top_token_ids"].shape == (2, 8, 4)


def test_hf_mode_local_files_only_wins_over_downloads(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="local-files-only"):
        build_teacher_textbook(
            TeacherTextbookBuildConfig(
                output_dir=tmp_path / "teacher_textbook",
                teacher_mode="hf",
                teacher_model_id="local/tiny-hf",
                local_files_only=True,
                allow_downloads=True,
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


def _install_mock_hf_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    calls: dict[str, list[bool]],
) -> None:
    class FakeTokenizer:
        name_or_path = "local/tiny-hf"
        vocab_size = 11
        pad_token_id = None
        eos_token = "<eos>"
        eos_token_id = 10
        bos_token_id = 9
        unk_token_id = 8

        def __call__(
            self,
            texts,
            *,
            padding,
            truncation,
            max_length,
            return_tensors,
        ):
            assert padding == "max_length"
            assert truncation is True
            assert return_tensors == "pt"
            input_ids = np.zeros((len(texts), max_length), dtype=np.int64)
            attention_mask = np.zeros_like(input_ids)
            for row, text in enumerate(texts):
                encoded = [((ord(ch) % 10) + 1) for ch in text[:max_length]]
                input_ids[row, : len(encoded)] = encoded
                attention_mask[row, : len(encoded)] = 1
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, *, local_files_only):
            assert model_id == "local/tiny-hf"
            calls["tokenizer"].append(local_files_only)
            return FakeTokenizer()

    class FakeModel:
        def eval(self):
            self.eval_called = True

        def __call__(self, **encoded):
            input_ids = np.asarray(encoded["input_ids"])
            batch, time = input_ids.shape
            logits = np.zeros((batch, time, 11), dtype=np.float32)
            for token in range(11):
                logits[:, :, token] = token + input_ids
            return SimpleNamespace(logits=logits)

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id, *, local_files_only):
            assert model_id == "local/tiny-hf"
            calls["model"].append(local_files_only)
            return FakeModel()

    class _InferenceMode:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False

    fake_torch = SimpleNamespace(inference_mode=lambda: _InferenceMode())
    fake_transformers = SimpleNamespace(
        AutoTokenizer=FakeAutoTokenizer,
        AutoModelForCausalLM=FakeAutoModel,
    )

    def fake_import_module(name: str):
        if name == "torch":
            return fake_torch
        if name == "transformers":
            return fake_transformers
        raise ImportError(name)

    monkeypatch.setattr(
        "qrwkv_xla.artifacts.teacher_textbook_builder.import_module",
        fake_import_module,
    )
