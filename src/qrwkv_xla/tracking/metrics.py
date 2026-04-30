from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricRecord:
    step: int
    values: dict[str, float]
    phase: str = "train"
    extra: dict[str, Any] = field(default_factory=dict)
    timestamp_utc: str = field(
        default_factory=lambda: (
            datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        )
    )


class MetricsLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def log(
        self,
        record: MetricRecord | None = None,
        *,
        step: int | None = None,
        values: dict[str, float] | None = None,
        phase: str = "train",
        extra: dict[str, Any] | None = None,
    ) -> MetricRecord:
        if record is None:
            if step is None or values is None:
                raise ValueError(
                    "step and values are required when record is not provided"
                )
            record = MetricRecord(
                step=int(step),
                values={str(key): float(value) for key, value in values.items()},
                phase=phase,
                extra=dict(extra or {}),
            )
        payload = asdict(record)
        payload["values"] = {
            str(key): float(value) for key, value in record.values.items()
        }
        self._handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._handle.flush()
        return record

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> MetricsLogger:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
