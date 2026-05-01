from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from qrwkv_xla.teacher_export import (
    ExportRequest,
    FakeTeacherExporter,
    TeacherExportConfig,
    load_teacher_export_config,
    resolve_prompts,
    validate_teacher_export_config,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_teacher_targets.py"


def test_corpus_configs_load() -> None:
    hf_config = load_teacher_export_config(
        ROOT / "configs" / "teacher_export_hf_tiny_corpus.yaml"
    )
    qwen_config = load_teacher_export_config(
        ROOT / "configs" / "teacher_export_qwen_dryrun_corpus.yaml"
    )

    assert hf_config.targets.prompt_corpus == ROOT / "corpora" / "smoke_prompts.jsonl"
    assert hf_config.targets.prompt_limit == 4
    assert qwen_config.targets.prompt_split == "train"


def test_prompt_loader_returns_corpus_metadata() -> None:
    config = load_teacher_export_config(
        ROOT / "configs" / "teacher_export_qwen_dryrun_corpus.yaml"
    )
    prompts = resolve_prompts(config)

    assert len(prompts.texts) == 4
    assert prompts.metadata["type"] == "corpus"
    assert prompts.metadata["corpus_id"] == "smoke_prompts"
    assert prompts.metadata["prompt_split"] == "train"
    assert prompts.metadata["prompt_limit"] == 4
    assert prompts.metadata["prompt_ids"] == [
        "smoke_000001",
        "smoke_000002",
        "smoke_000003",
        "smoke_000004",
    ]


def test_config_rejects_ambiguous_prompt_sources() -> None:
    config = TeacherExportConfig()
    config = replace(
        config,
        targets=replace(
            config.targets,
            prompt_texts=("inline",),
            prompt_corpus=ROOT / "corpora" / "smoke_prompts.jsonl",
        ),
    )

    with pytest.raises(
        ValueError,
        match="Use either prompt_texts/prompt_file or prompt_corpus",
    ):
        validate_teacher_export_config(config)


def test_qwen_dry_run_corpus_prints_prompt_source_without_hf_import() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(ROOT / "configs" / "teacher_export_qwen_dryrun_corpus.yaml"),
            "--dry-run",
            "--resolve-qwen-policy",
            "--allow-unresolved-policy",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "prompt_source_type: corpus" in result.stdout
    assert '"corpus_id": "smoke_prompts"' in result.stdout


def test_fake_exporter_records_prompt_source(tmp_path: Path) -> None:
    config = load_teacher_export_config(
        ROOT / "configs" / "teacher_export_qwen_dryrun_corpus.yaml"
    )
    config = replace(
        config,
        targets=replace(config.targets, hidden_size=4, num_layers=1, vocab_size=8),
        runtime=replace(
            config.runtime,
            exporter_backend="fake",
            require_resolved_model=False,
            output_dir=tmp_path / "bundle",
        ),
    )

    result = FakeTeacherExporter().export(
        ExportRequest(config=config, output_dir=config.runtime.output_dir)
    )

    assert result.manifest.prompt_source is not None
    assert result.manifest.prompt_source["type"] == "corpus"
    assert result.manifest.prompt_source["prompt_count"] == 4
    assert "texts" not in result.manifest.prompt_source
