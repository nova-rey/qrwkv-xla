from __future__ import annotations

import json
from pathlib import Path

from qrwkv_xla.tracking import MetricRecord, MetricsLogger


def test_metrics_logger_writes_jsonl_and_flushes(tmp_path: Path) -> None:
    metrics_path = tmp_path / "runs" / "unit" / "metrics.jsonl"
    with MetricsLogger(metrics_path) as logger:
        record = logger.log(
            step=1,
            values={"loss": 1.25, "hidden_mse": 1.25},
            extra={"stage": 0},
        )

    assert record.step == 1
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["step"] == 1
    assert payload["phase"] == "train"
    assert payload["values"]["loss"] == 1.25
    assert payload["extra"]["stage"] == 0
    assert "timestamp_utc" in payload


def test_metrics_logger_appends_one_object_per_line(tmp_path: Path) -> None:
    metrics_path = tmp_path / "runs" / "unit" / "metrics.jsonl"
    with MetricsLogger(metrics_path) as logger:
        logger.log(step=1, values={"loss": 2.0})
        logger.log(step=2, values={"loss": 1.0})

    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["step"] for record in records] == [1, 2]
    assert [record["values"]["loss"] for record in records] == [2.0, 1.0]


def test_metrics_logger_accepts_metric_record(tmp_path: Path) -> None:
    metrics_path = tmp_path / "runs" / "unit" / "metrics.jsonl"
    with MetricsLogger(metrics_path) as logger:
        logger.log(MetricRecord(step=3, values={"loss": 0.5}, extra={"local_step": 1}))

    payload = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["step"] == 3
    assert payload["extra"]["local_step"] == 1
