from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from qrwkv_xla.artifacts import (
    load_fingerprint_exemplars,
    load_fingerprint_targets,
    summarize_fingerprint_artifact,
    validate_fingerprint_artifact,
)
from qrwkv_xla.fingerprint import (
    DEFAULT_TINY_REAL_TEACHER,
    TARGET_PAYLOAD_PACKED_CORRIDOR_V1,
    CaptureTopologyConfig,
    FingerprintCaptureBudgetConfig,
    FingerprintCaptureConfig,
    FingerprintCaptureProgressConfig,
    TinyRealTeacherFingerprintCaptureConfig,
    build_synthetic_capture_examples,
    capture_fingerprint_artifact,
    detect_cpu_budget,
    load_text_fixture,
    resolve_capture_topology,
    run_tiny_real_teacher_fingerprint_capture,
)
from qrwkv_xla.teachers import HFTeacherBackend

ROOT = Path(__file__).resolve().parents[1]
TEXTS = (
    ROOT
    / "tests"
    / "fixtures"
    / "fingerprint_capture_real_teacher"
    / "tiny_texts.jsonl"
)
SCRIPT = ROOT / "scripts" / "build_real_teacher_fingerprint_artifact.py"


def test_tiny_real_teacher_capture_smoke_with_fake_hf_backend(tmp_path: Path) -> None:
    result = _run_fake_capture(tmp_path, max_exemplars=5)
    summary = _json(result.summary_path)

    assert result.status == "pass"
    assert result.artifact_validated is True
    assert result.teacher_real is True
    assert result.teacher_backend == "hf_causal_lm"
    assert result.local_files_only is True
    assert result.examples_processed > 0
    assert result.target_positions_processed > 0
    assert result.modes_discovered > 0
    assert result.exemplars_retained <= 5
    assert summary["phase"] == "P145"
    assert summary["capture_engine"] == "teacher_side_capture_skeleton_v0"
    assert summary["run_kind"] == "tiny_real_teacher_capture"


def test_artifact_validates_loads_and_summarizes(tmp_path: Path) -> None:
    result = _run_fake_capture(tmp_path, max_exemplars=3)
    validation = validate_fingerprint_artifact(result.output_dir)
    targets = load_fingerprint_targets(result.output_dir, batch_size=2)
    exemplars = load_fingerprint_exemplars(result.output_dir, batch_size=2)
    artifact = summarize_fingerprint_artifact(result.output_dir)

    assert validation.ok is True
    assert targets.num_records == result.target_positions_processed
    assert exemplars.num_records == result.exemplars_retained
    assert artifact.num_corridor_records == result.target_positions_processed
    assert artifact.has_exemplars is True
    assert artifact.num_exemplar_records == result.exemplars_retained


def test_teacher_metadata_is_present_in_manifest_and_summary(tmp_path: Path) -> None:
    result = _run_fake_capture(tmp_path, max_exemplars=2)
    manifest = _json(result.manifest_path)
    summary = _json(result.summary_path)

    assert manifest["teacher"]["backend"] == "hf_causal_lm"
    assert manifest["teacher"]["model_name_or_path"] == "local/fake-real-teacher"
    assert manifest["teacher"]["tokenizer_name_or_path"] == "local/fake-tokenizer"
    assert manifest["teacher"]["local_files_only"] is True
    assert manifest["teacher"]["teacher_real"] is True
    assert manifest["teacher"]["vocab_size"] == 16
    assert summary["teacher_real"] is True
    assert summary["teacher_backend"] == "hf_causal_lm"
    assert summary["teacher"]["vocab_size"] == 16
    assert summary["local_files_only"] is True


def test_summary_fields_and_consumer_sanity_are_recorded(tmp_path: Path) -> None:
    result = _run_fake_capture(tmp_path, max_exemplars=4)
    summary = _json(result.summary_path)

    assert summary["examples_processed"] == 4
    assert summary["tokens_processed"] > 0
    assert summary["target_positions_processed"] == 16
    assert summary["positions_policy"] == "fixed_all_positions"
    assert summary["modes_discovered"] > 0
    assert summary["records_per_mode"]
    assert summary["corridor_bounds_method"] == "quantile"
    assert summary["exemplar_selection_policy"] == "stratified_interestingness_v0"
    assert summary["max_exemplars"] == 4
    assert summary["exemplars_retained"] <= 4
    assert summary["artifact_validated"] is True
    assert summary["targets_loadable"] is True
    assert summary["exemplars_loadable"] is True
    assert summary["consumer_sanity"]["kind"] in {
        "compressed_exemplar_optimizer_step",
        "p141_one_step",
        "p140_forward",
        "loader_only",
    }
    assert summary["consumer_sanity"]["status"] == "pass"
    if summary["consumer_sanity"]["kind"] == "compressed_exemplar_optimizer_step":
        assert summary["consumer_sanity"]["loss_finite"] is True
        assert summary["consumer_sanity"]["gradient_finite"] is True
        assert summary["consumer_sanity"]["gradient_norm"] > 0.0
        assert summary["consumer_sanity"]["gradient_norm_finite"] is True
        assert summary["consumer_sanity"]["gradient_norm_positive"] is True
        assert summary["consumer_sanity"]["parameters_changed"] is True


def test_exemplar_budget_is_honored(tmp_path: Path) -> None:
    result = _run_fake_capture(tmp_path, max_exemplars=3)
    manifest = _json(result.manifest_path)
    summary = _json(result.summary_path)

    assert result.exemplars_retained <= 3
    assert manifest["exemplar_reservoir"]["num_records"] <= 3
    assert summary["max_exemplars"] == 3
    assert summary["exemplars_retained"] <= 3


def test_large_vocab_records_loader_only_consumer_reason(tmp_path: Path) -> None:
    result = run_tiny_real_teacher_fingerprint_capture(
        TinyRealTeacherFingerprintCaptureConfig(
            output_dir=tmp_path / "artifact",
            texts_path=TEXTS,
            sequence_length=4,
            max_examples=2,
            max_target_positions=8,
            overwrite=True,
            max_exemplars=2,
            exemplar_target_type="dense_probs",
            consumer_vocab_limit=8,
        ),
        backend=_fake_backend(vocab_size=16),
    )
    summary = _json(result.summary_path)

    assert summary["consumer_sanity"]["kind"] == "loader_only"
    assert summary["consumer_sanity"]["status"] == "pass"
    assert "vocab too large" in summary["consumer_sanity"]["reason"]


def test_capture_progress_json_is_atomic_monotonic_and_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qrwkv_xla.fingerprint import capture as capture_module

    ticks = iter(float(value) for value in range(100))
    monkeypatch.setattr(capture_module.time, "monotonic", lambda: next(ticks))
    real_replace = capture_module.os.replace
    writes: list[dict[str, Any]] = []

    def recording_replace(src: object, dst: object) -> None:
        writes.append(json.loads(Path(src).read_text(encoding="utf-8")))
        real_replace(src, dst)

    monkeypatch.setattr(capture_module.os, "replace", recording_replace)
    progress_path = tmp_path / "artifact" / "progress.json"
    examples = build_synthetic_capture_examples(
        num_examples=3,
        max_seq_len=4,
        vocab_size=16,
    )

    result = capture_fingerprint_artifact(
        FingerprintCaptureConfig(
            output_dir=tmp_path / "artifact",
            overwrite=True,
            capture_budget=FingerprintCaptureBudgetConfig(max_target_positions=12),
            target_payload_type=TARGET_PAYLOAD_PACKED_CORRIDOR_V1,
            progress=FingerprintCaptureProgressConfig(
                enabled=True,
                progress_path=progress_path,
                interval_seconds=999.0,
                interval_examples=1,
            ),
        ),
        examples,
    )
    final = _json(progress_path)

    assert result.validation_ok is True
    assert writes[0]["status"] == "running"
    assert writes[0]["eta_seconds"] is None
    assert any(
        payload["status"] == "running" and payload["eta_seconds"] is not None
        for payload in writes
    )
    assert writes[-1]["status"] == "complete"
    assert final["status"] == "complete"
    assert final["artifact_validated"] is True
    assert final["artifact_size_bytes"] > 0
    assert final["examples_processed"] == 3
    assert final["target_positions_processed"] == 12
    assert final["target_capture_memory_kind"] == "preallocated_typed_arrays"
    assert [payload["examples_processed"] for payload in writes] == sorted(
        payload["examples_processed"] for payload in writes
    )
    assert [payload["target_positions_processed"] for payload in writes] == sorted(
        payload["target_positions_processed"] for payload in writes
    )


def test_capture_progress_failure_status_preserves_exception(
    tmp_path: Path,
) -> None:
    progress_path = tmp_path / "artifact" / "progress.json"
    valid = build_synthetic_capture_examples(
        num_examples=1,
        max_seq_len=4,
        vocab_size=16,
    )[0]
    invalid = build_synthetic_capture_examples(
        num_examples=1,
        max_seq_len=5,
        vocab_size=16,
    )[0]

    with pytest.raises(ValueError, match="share logits shape"):
        capture_fingerprint_artifact(
            FingerprintCaptureConfig(
                output_dir=tmp_path / "artifact",
                overwrite=True,
                capture_budget=FingerprintCaptureBudgetConfig(max_target_positions=8),
                progress=FingerprintCaptureProgressConfig(
                    enabled=True,
                    progress_path=progress_path,
                    interval_seconds=999.0,
                    interval_examples=1,
                ),
            ),
            (valid, invalid),
        )

    progress = _json(progress_path)
    assert progress["status"] == "failed"
    assert progress["exception_type"] == "ValueError"
    assert "share logits shape" in progress["exception_message"]


def test_real_teacher_progress_defaults_to_output_dir_complete(
    tmp_path: Path,
) -> None:
    result = run_tiny_real_teacher_fingerprint_capture(
        TinyRealTeacherFingerprintCaptureConfig(
            output_dir=tmp_path / "artifact",
            texts_path=TEXTS,
            sequence_length=4,
            max_examples=4,
            max_target_positions=16,
            overwrite=True,
            max_exemplars=3,
            consumer_vocab_limit=64,
            progress_interval_examples=1,
        ),
        backend=_fake_backend(vocab_size=16),
    )
    progress = _json(result.output_dir / "progress.json")

    assert progress["status"] == "complete"
    assert progress["phase"] == "teacher_capture"
    assert progress["examples_processed"] == result.examples_processed
    assert progress["target_positions_processed"] == result.target_positions_processed
    assert progress["modes_discovered"] == result.modes_discovered
    assert progress["exemplars_retained"] == result.exemplars_retained
    assert progress["consumer_sanity"]["status"] == "pass"
    assert progress["teacher_model"] == DEFAULT_TINY_REAL_TEACHER
    assert progress["pid"] > 0


@pytest.mark.parametrize("teacher_batch_size", [1, 2, 4, 8])
def test_real_teacher_microbatch_matches_batch_one_artifact(
    tmp_path: Path,
    teacher_batch_size: int,
) -> None:
    reference = _run_fake_capture(
        tmp_path / "ref",
        max_exemplars=5,
        teacher_batch_size=1,
    )
    candidate = _run_fake_capture(
        tmp_path / "candidate",
        max_exemplars=5,
        teacher_batch_size=teacher_batch_size,
    )

    assert (
        _json(reference.manifest_path)["target_payload"]
        == _json(candidate.manifest_path)["target_payload"]
    )
    assert _json(reference.output_dir / "modes.json") == _json(
        candidate.output_dir / "modes.json"
    )
    assert _target_arrays(reference.output_dir) == _target_arrays(candidate.output_dir)
    assert _jsonl(reference.output_dir / "exemplars" / "exemplars-00000.jsonl") == (
        _jsonl(candidate.output_dir / "exemplars" / "exemplars-00000.jsonl")
    )
    assert candidate.target_positions_processed == reference.target_positions_processed


def test_real_teacher_microbatch_handles_incomplete_final_batch(
    tmp_path: Path,
) -> None:
    backend = _fake_backend(vocab_size=16)
    result = run_tiny_real_teacher_fingerprint_capture(
        TinyRealTeacherFingerprintCaptureConfig(
            output_dir=tmp_path / "artifact",
            texts_path=TEXTS,
            sequence_length=4,
            max_examples=3,
            max_target_positions=12,
            overwrite=True,
            max_exemplars=3,
            consumer_vocab_limit=64,
            progress_interval_examples=1,
            teacher_batch_size=2,
        ),
        backend=backend,
    )
    summary = _json(result.summary_path)
    progress = _json(result.output_dir / "progress.json")

    assert backend.model.call_batch_sizes == [2, 1]
    assert result.examples_processed == 3
    assert result.target_positions_processed == 12
    assert summary["teacher_batch_size"] == 2
    assert summary["batches_processed"] == 2
    assert summary["effective_examples_per_batch"] == pytest.approx(1.5)
    assert progress["teacher_batch_size"] == 2
    assert progress["batches_processed"] == 2
    assert progress["effective_examples_per_batch"] == pytest.approx(1.5)


def test_real_teacher_staged_prefetch_tokenizes_before_inference(
    tmp_path: Path,
) -> None:
    backend = _fake_backend(vocab_size=16)
    tokenizer = backend.tokenizer
    assert isinstance(tokenizer, _FakeTokenizer)
    runtime_budget = _runtime_cpu_budget()

    result = run_tiny_real_teacher_fingerprint_capture(
        TinyRealTeacherFingerprintCaptureConfig(
            output_dir=tmp_path / "artifact",
            texts_path=TEXTS,
            sequence_length=4,
            max_examples=4,
            max_target_positions=16,
            overwrite=True,
            max_exemplars=3,
            consumer_vocab_limit=64,
            progress_interval_examples=1,
            teacher_batch_size=2,
            cpu_budget=runtime_budget,
            torch_num_threads=1,
            prefetch_workers=1,
            prefetch_depth=1,
            result_queue_depth=1,
        ),
        backend=backend,
    )
    summary = _json(result.summary_path)

    assert tokenizer.call_count == 2
    assert backend.model.call_batch_sizes == [2, 2]
    assert summary["prepared_batches"] == 2
    assert summary["inference_batches"] == 2
    assert summary["reduced_batches"] is None
    assert summary["committed_batches"] == 2


def test_real_teacher_staged_multiple_prefetch_workers_preserve_order(
    tmp_path: Path,
) -> None:
    runtime_budget = _runtime_cpu_budget(minimum=4)
    reference = _run_fake_capture(
        tmp_path / "sync",
        max_exemplars=5,
        teacher_batch_size=2,
        cpu_budget=runtime_budget,
        torch_num_threads=1,
        prefetch_workers=0,
        result_queue_depth=1,
    )
    candidate = _run_fake_capture(
        tmp_path / "multi",
        max_exemplars=5,
        teacher_batch_size=2,
        cpu_budget=runtime_budget,
        torch_num_threads=1,
        prefetch_workers=2,
        prefetch_depth=2,
        result_queue_depth=1,
    )

    assert _target_arrays(reference.output_dir) == _target_arrays(candidate.output_dir)


def test_real_teacher_rejects_invalid_teacher_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="teacher_batch_size"):
        run_tiny_real_teacher_fingerprint_capture(
            TinyRealTeacherFingerprintCaptureConfig(
                output_dir=tmp_path / "artifact",
                texts_path=TEXTS,
                teacher_batch_size=3,
            ),
            backend=_fake_backend(vocab_size=16),
        )


@pytest.mark.parametrize("detected_cpu_budget", [1, 2, 3, 4, 8, 12])
def test_capture_topology_auto_policy_simulated_cpu_budgets(
    detected_cpu_budget: int,
) -> None:
    topology = resolve_capture_topology(
        CaptureTopologyConfig(teacher_device="cpu"),
        detected_cpu_budget=detected_cpu_budget,
    )
    runnable = (
        topology.torch_num_threads
        + topology.prefetch_workers
        + topology.reducer_workers
        + 1
    )

    assert topology.teacher_device == "cpu"
    assert topology.detected_cpu_budget == detected_cpu_budget
    assert 1 <= topology.effective_cpu_budget <= detected_cpu_budget
    assert topology.torch_num_threads >= 1
    assert topology.torch_num_interop_threads == 1
    assert topology.prefetch_workers >= 0
    assert topology.reducer_workers >= 0
    assert runnable <= max(2, topology.effective_cpu_budget + 1)


def test_capture_topology_rejects_explicit_oversubscription() -> None:
    with pytest.raises(ValueError, match="oversubscribes"):
        resolve_capture_topology(
            CaptureTopologyConfig(
                cpu_budget=4,
                torch_num_threads=4,
                prefetch_workers=2,
                reducer_workers=0,
                teacher_device="cpu",
            ),
            detected_cpu_budget=4,
        )


def test_capture_topology_rejects_nonzero_reducer_workers() -> None:
    with pytest.raises(ValueError, match="reducer_workers>0 is deferred"):
        resolve_capture_topology(
            CaptureTopologyConfig(
                cpu_budget=4,
                torch_num_threads=1,
                prefetch_workers=1,
                reducer_workers=1,
                teacher_device="cpu",
            ),
            detected_cpu_budget=4,
        )


def test_capture_topology_records_gpu_reduction_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qrwkv_xla.fingerprint import topology as topology_module

    monkeypatch.setattr(
        topology_module,
        "_device_available",
        lambda device: device in {"cpu", "cuda"},
    )
    monkeypatch.setattr(topology_module, "_gpu_name", lambda device: f"fake-{device}")

    topology = topology_module.resolve_capture_topology(
        topology_module.CaptureTopologyConfig(
            cpu_budget=4,
            torch_num_threads=1,
            prefetch_workers=1,
            teacher_device="cuda",
        ),
        detected_cpu_budget=4,
    )

    assert topology.teacher_device == "cuda"
    assert topology.gpu_name == "fake-cuda"
    assert topology.gpu_reduction_mode == "full_logits_host_transfer_fallback"


def test_real_teacher_staged_pipeline_matches_sync_artifact(tmp_path: Path) -> None:
    runtime_budget = _runtime_cpu_budget()
    reference = _run_fake_capture(
        tmp_path / "sync",
        max_exemplars=5,
        teacher_batch_size=2,
        cpu_budget=runtime_budget,
        torch_num_threads=1,
        prefetch_workers=0,
        result_queue_depth=1,
    )
    candidate = _run_fake_capture(
        tmp_path / "staged",
        max_exemplars=5,
        teacher_batch_size=2,
        cpu_budget=runtime_budget,
        torch_num_threads=1,
        prefetch_workers=1,
        prefetch_depth=1,
        result_queue_depth=1,
    )

    assert _json(reference.output_dir / "modes.json") == _json(
        candidate.output_dir / "modes.json"
    )
    assert _target_arrays(reference.output_dir) == _target_arrays(candidate.output_dir)
    assert _jsonl(reference.output_dir / "exemplars" / "exemplars-00000.jsonl") == (
        _jsonl(candidate.output_dir / "exemplars" / "exemplars-00000.jsonl")
    )


def test_real_teacher_progress_records_topology_and_staged_counters(
    tmp_path: Path,
) -> None:
    runtime_budget = _runtime_cpu_budget()
    result = _run_fake_capture(
        tmp_path,
        max_exemplars=3,
        teacher_batch_size=2,
        cpu_budget=runtime_budget,
        torch_num_threads=1,
        prefetch_workers=1,
        prefetch_depth=1,
        result_queue_depth=1,
    )
    summary = _json(result.summary_path)
    progress = _json(result.output_dir / "progress.json")

    for payload in (summary, progress):
        assert payload["teacher_device"] == "cpu"
        assert payload["detected_cpu_budget"] >= 1
        assert payload["effective_cpu_budget"] == runtime_budget
        assert payload["torch_num_threads"] == 1
        assert payload["prefetch_workers"] == 1
        assert payload["prefetch_depth"] == 1
        assert payload["result_queue_depth"] == 1
        assert payload["prepared_batches"] == 2
        assert payload["inference_batches"] == 2
        assert payload["reduced_batches"] is None
        assert payload["committed_batches"] == 2
    assert summary["capture_topology"]["teacher_device"] == "cpu"


def test_real_teacher_manifest_records_resolved_device_and_transfer_metadata(
    tmp_path: Path,
) -> None:
    result = _run_fake_capture(
        tmp_path,
        max_exemplars=3,
        teacher_batch_size=2,
        cpu_budget=_runtime_cpu_budget(),
        torch_num_threads=1,
        prefetch_workers=1,
        prefetch_depth=1,
        result_queue_depth=1,
    )
    manifest = _json(result.manifest_path)
    summary = _json(result.summary_path)
    progress = _json(result.output_dir / "progress.json")

    assert manifest["teacher"]["device"] == "cpu"
    assert manifest["teacher"]["pin_memory"] is False
    assert manifest["teacher"]["non_blocking_transfer"] is False
    assert manifest["teacher"]["gpu_name"] is None
    assert manifest["teacher"]["gpu_reduction_mode"] == "not_applicable"
    assert summary["teacher"]["device"] == manifest["teacher"]["device"]
    assert summary["gpu_reduction_mode"] == "not_applicable"
    assert progress["gpu_reduction_mode"] == "not_applicable"


def test_real_teacher_staged_prefetch_failure_marks_progress_failed(
    tmp_path: Path,
) -> None:
    backend = _fake_backend(vocab_size=16, fail_on_tokenize=1)
    progress_path = tmp_path / "artifact" / "progress.json"
    runtime_budget = _runtime_cpu_budget()

    with pytest.raises(RuntimeError, match="prefetch stage failed: RuntimeError"):
        run_tiny_real_teacher_fingerprint_capture(
            TinyRealTeacherFingerprintCaptureConfig(
                output_dir=tmp_path / "artifact",
                texts_path=TEXTS,
                sequence_length=4,
                max_examples=4,
                max_target_positions=16,
                overwrite=True,
                max_exemplars=3,
                consumer_vocab_limit=64,
                progress_interval_examples=1,
                teacher_batch_size=2,
                cpu_budget=runtime_budget,
                torch_num_threads=1,
                prefetch_workers=1,
                prefetch_depth=1,
                result_queue_depth=1,
            ),
            backend=backend,
        )
    progress = _json(progress_path)

    assert progress["status"] == "failed"
    assert progress["exception_type"] == "RuntimeError"
    assert "synthetic tokenizer failure" in progress["exception_message"]


def test_real_teacher_staged_inference_failure_marks_progress_failed(
    tmp_path: Path,
) -> None:
    backend = _fake_backend(vocab_size=16, fail_on_call=2)
    progress_path = tmp_path / "artifact" / "progress.json"
    runtime_budget = _runtime_cpu_budget()

    with pytest.raises(RuntimeError, match="inference stage failed: RuntimeError"):
        run_tiny_real_teacher_fingerprint_capture(
            TinyRealTeacherFingerprintCaptureConfig(
                output_dir=tmp_path / "artifact",
                texts_path=TEXTS,
                sequence_length=4,
                max_examples=4,
                max_target_positions=16,
                overwrite=True,
                max_exemplars=3,
                consumer_vocab_limit=64,
                progress_interval_examples=1,
                teacher_batch_size=2,
                cpu_budget=runtime_budget,
                torch_num_threads=1,
                prefetch_workers=1,
                prefetch_depth=1,
                result_queue_depth=1,
            ),
            backend=backend,
        )
    progress = _json(progress_path)

    assert progress["status"] == "failed"
    assert progress["exception_type"] == "RuntimeError"
    assert "synthetic teacher failure" in progress["exception_message"]


def test_project_active_files_do_not_set_deprecated_transformers_cache() -> None:
    completed = subprocess.run(
        [
            "git",
            "grep",
            "-n",
            "TRANSFORMERS_CACHE",
            "--",
            "src",
            "scripts",
            ".github",
            "configs",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1, completed.stdout


def test_text_fixture_loads_expected_rows() -> None:
    texts = load_text_fixture(TEXTS)

    assert len(texts) == 4
    assert texts[0].startswith("The quick brown fox")


def test_config_defaults_to_local_files_only(tmp_path: Path) -> None:
    config = TinyRealTeacherFingerprintCaptureConfig(
        output_dir=tmp_path / "artifact",
        texts_path=TEXTS,
    )

    assert config.local_files_only is True
    assert config.allow_downloads is False
    assert config.exemplar_target_type == "cascaded_soft_labels_v1"
    assert config.exemplar_top_k == 256


def test_optional_cli_smoke_skips_when_local_teacher_unavailable(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "cli_artifact"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--teacher-model",
            DEFAULT_TINY_REAL_TEACHER,
            "--texts",
            str(TEXTS),
            "--output-dir",
            str(output_dir),
            "--sequence-length",
            "8",
            "--max-examples",
            "2",
            "--max-target-positions",
            "8",
            "--max-exemplars",
            "2",
            "--local-files-only",
            "--overwrite",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode == 2:
        pytest.skip("tiny HF teacher not available in local cache")
    assert completed.returncode == 0, completed.stderr
    assert "status=pass" in completed.stdout
    assert validate_fingerprint_artifact(output_dir).ok is True


def test_p145_docs_do_not_claim_quality_improvement() -> None:
    doc = ROOT / "docs" / "P145_TINY_REAL_TEACHER_FINGERPRINT_CAPTURE.md"
    if not doc.exists():
        pytest.skip("P145 docs not written yet")
    text = doc.read_text(encoding="utf-8").lower()

    assert "quality improvement" in text
    assert "does not prove" in text
    assert "baseline" in text


def _run_fake_capture(
    tmp_path: Path,
    *,
    max_exemplars: int,
    teacher_batch_size: int = 1,
    cpu_budget: int | str = "auto",
    torch_num_threads: int | str = "auto",
    prefetch_workers: int | str = "auto",
    prefetch_depth: int = 2,
    result_queue_depth: int = 2,
) -> object:
    return run_tiny_real_teacher_fingerprint_capture(
        TinyRealTeacherFingerprintCaptureConfig(
            output_dir=tmp_path / "artifact",
            texts_path=TEXTS,
            sequence_length=4,
            max_examples=4,
            max_target_positions=16,
            overwrite=True,
            max_exemplars=max_exemplars,
            consumer_vocab_limit=64,
            teacher_batch_size=teacher_batch_size,
            cpu_budget=cpu_budget,
            torch_num_threads=torch_num_threads,
            prefetch_workers=prefetch_workers,
            prefetch_depth=prefetch_depth,
            result_queue_depth=result_queue_depth,
            teacher_device="cpu",
        ),
        backend=_fake_backend(vocab_size=16),
    )


def _fake_backend(
    *,
    vocab_size: int,
    fail_on_call: int | None = None,
    fail_on_tokenize: int | None = None,
) -> HFTeacherBackend:
    return HFTeacherBackend(
        "local/fake-real-teacher",
        tokenizer=_FakeTokenizer(vocab_size=vocab_size, fail_on_call=fail_on_tokenize),
        model=_FakeCausalLM(vocab_size=vocab_size, fail_on_call=fail_on_call),
        prompts=("placeholder",),
    )


class _FakeTokenizer:
    name_or_path = "local/fake-tokenizer"
    eos_token = "<eos>"
    pad_token = None
    eos_token_id = 1
    pad_token_id = 1
    bos_token_id = 0
    unk_token_id = 2

    def __init__(self, *, vocab_size: int, fail_on_call: int | None = None) -> None:
        self.vocab_size = vocab_size
        self.fail_on_call = fail_on_call
        self.call_count = 0

    def __call__(
        self,
        prompts: list[str],
        *,
        padding: str,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, np.ndarray]:
        self.call_count += 1
        if self.fail_on_call is not None and self.call_count >= self.fail_on_call:
            raise RuntimeError("synthetic tokenizer failure")
        assert padding == "max_length"
        assert truncation is True
        assert return_tensors == "pt"
        rows = []
        masks = []
        for prompt in prompts:
            prompt_len = min(max(1, len(prompt.split())), max_length)
            prompt_seed = sum(ord(char) for char in prompt) % self.vocab_size
            rows.append(
                [(prompt_seed + col) % self.vocab_size for col in range(max_length)]
            )
            masks.append([1 if col < prompt_len else 0 for col in range(max_length)])
        return {
            "input_ids": np.asarray(rows, dtype=np.int32),
            "attention_mask": np.asarray(masks, dtype=np.int32),
        }


class _FakeCausalLM:
    def __init__(self, *, vocab_size: int, fail_on_call: int | None = None) -> None:
        self.vocab_size = vocab_size
        self.fail_on_call = fail_on_call
        self.call_batch_sizes: list[int] = []

    def eval(self) -> None:
        return None

    def __call__(self, *, input_ids: Any, attention_mask: Any | None) -> object:
        del attention_mask
        ids = np.asarray(input_ids, dtype=np.float32)
        self.call_batch_sizes.append(int(ids.shape[0]))
        if (
            self.fail_on_call is not None
            and len(self.call_batch_sizes) >= self.fail_on_call
        ):
            raise RuntimeError("synthetic teacher failure")
        vocab = np.arange(self.vocab_size, dtype=np.float32)[None, None, :]
        logits = np.sin(ids[:, :, None] * 0.17 + vocab * 0.23)
        return SimpleNamespace(logits=logits.astype(np.float32))


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _target_arrays(artifact_dir: Path) -> dict[str, list]:
    return {
        path.name: np.load(path, allow_pickle=False).tolist()
        for path in sorted((artifact_dir / "targets").glob("*.npy"))
    }


def _runtime_cpu_budget(*, minimum: int = 2) -> int:
    detected = detect_cpu_budget()
    if detected < minimum:
        pytest.skip(f"staged runtime test needs at least {minimum} detected CPUs")
    return min(detected, minimum)
