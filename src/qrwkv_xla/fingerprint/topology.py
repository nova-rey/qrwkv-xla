from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Literal

AutoInt = int | Literal["auto"]
AutoBool = bool | Literal["auto"]

SUPPORTED_TEACHER_DEVICES = ("auto", "cpu", "cuda", "mps")


@dataclass(frozen=True)
class CaptureTopologyConfig:
    cpu_budget: AutoInt = "auto"
    torch_num_threads: AutoInt = "auto"
    torch_num_interop_threads: AutoInt = "auto"
    prefetch_workers: AutoInt = "auto"
    reducer_workers: AutoInt = "auto"
    prefetch_depth: int = 2
    result_queue_depth: int = 2
    teacher_device: str = "auto"
    pin_memory: AutoBool = "auto"
    non_blocking_transfer: AutoBool = "auto"


@dataclass(frozen=True)
class ResolvedCaptureTopology:
    teacher_device: str
    detected_cpu_budget: int
    effective_cpu_budget: int
    torch_num_threads: int
    torch_num_interop_threads: int
    prefetch_workers: int
    reducer_workers: int
    prefetch_depth: int
    result_queue_depth: int
    pin_memory: bool
    non_blocking_transfer: bool
    omp_num_threads: str | None
    mkl_num_threads: str | None
    gpu_name: str | None = None
    gpu_reduction_mode: str = "not_applicable"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_capture_topology(
    config: CaptureTopologyConfig,
    *,
    detected_cpu_budget: int | None = None,
) -> ResolvedCaptureTopology:
    detected = int(detected_cpu_budget or detect_cpu_budget())
    if detected <= 0:
        detected = 1
    device = _resolve_teacher_device(config.teacher_device)
    effective = _resolve_auto_int(
        config.cpu_budget,
        name="cpu_budget",
        detected_cpu_budget=detected,
        default=_auto_effective_cpu_budget(detected),
        allow_zero=False,
    )
    torch_threads = _resolve_auto_int(
        config.torch_num_threads,
        name="torch_num_threads",
        detected_cpu_budget=effective,
        default=_auto_torch_threads(effective, teacher_device=device),
        allow_zero=False,
    )
    torch_interop = _resolve_auto_int(
        config.torch_num_interop_threads,
        name="torch_num_interop_threads",
        detected_cpu_budget=effective,
        default=1,
        allow_zero=False,
    )
    prefetch_workers = _resolve_auto_int(
        config.prefetch_workers,
        name="prefetch_workers",
        detected_cpu_budget=effective,
        default=_auto_prefetch_workers(effective, teacher_device=device),
        allow_zero=True,
    )
    reducer_workers = _resolve_reducer_workers(config.reducer_workers)
    _validate_depth(config.prefetch_depth, name="prefetch_depth")
    _validate_depth(config.result_queue_depth, name="result_queue_depth")
    runnable = torch_threads + prefetch_workers + reducer_workers + 1
    if runnable > max(2, effective + 1):
        raise ValueError(
            "topology worker/thread allocation oversubscribes cpu_budget: "
            f"runnable={runnable} effective_cpu_budget={effective}"
        )
    return ResolvedCaptureTopology(
        teacher_device=device,
        detected_cpu_budget=detected,
        effective_cpu_budget=effective,
        torch_num_threads=torch_threads,
        torch_num_interop_threads=torch_interop,
        prefetch_workers=prefetch_workers,
        reducer_workers=reducer_workers,
        prefetch_depth=int(config.prefetch_depth),
        result_queue_depth=int(config.result_queue_depth),
        pin_memory=_resolve_auto_bool(
            config.pin_memory,
            default=device in {"cuda", "mps"},
            name="pin_memory",
        ),
        non_blocking_transfer=_resolve_auto_bool(
            config.non_blocking_transfer,
            default=device in {"cuda", "mps"},
            name="non_blocking_transfer",
        ),
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        mkl_num_threads=os.environ.get("MKL_NUM_THREADS"),
        gpu_name=_gpu_name(device),
        gpu_reduction_mode=_gpu_reduction_mode(device),
    )


def detect_cpu_budget() -> int:
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            return max(1, len(affinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def apply_torch_thread_settings(topology: ResolvedCaptureTopology) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", str(topology.torch_num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(topology.torch_num_threads))
    result: dict[str, Any] = {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "torch_num_threads": topology.torch_num_threads,
        "torch_num_interop_threads": topology.torch_num_interop_threads,
        "torch_thread_settings_applied": False,
        "torch_thread_settings_error": None,
    }
    try:
        import torch
    except ImportError:
        return result
    try:
        torch.set_num_threads(topology.torch_num_threads)
        torch.set_num_interop_threads(topology.torch_num_interop_threads)
        result["torch_thread_settings_applied"] = True
    except RuntimeError as exc:
        result["torch_thread_settings_error"] = str(exc)
    return result


def _auto_effective_cpu_budget(detected: int) -> int:
    if detected <= 2:
        return detected
    if detected <= 4:
        return detected - 1
    return detected - 2


def _auto_torch_threads(effective: int, *, teacher_device: str) -> int:
    if teacher_device in {"cuda", "mps"}:
        return 1
    if effective <= 2:
        return 1
    if effective <= 4:
        return 2
    return max(2, effective - 2)


def _auto_prefetch_workers(effective: int, *, teacher_device: str) -> int:
    if effective <= 2:
        return 0
    if teacher_device in {"cuda", "mps"}:
        return min(2, effective - 1)
    return 1


def _resolve_auto_int(
    value: AutoInt,
    *,
    name: str,
    detected_cpu_budget: int,
    default: int,
    allow_zero: bool,
) -> int:
    if value == "auto":
        resolved = int(default)
    else:
        resolved = int(value)
    minimum = 0 if allow_zero else 1
    if resolved < minimum:
        raise ValueError(f"{name} must be >= {minimum} or 'auto'")
    if resolved > detected_cpu_budget:
        raise ValueError(
            f"{name}={resolved} exceeds effective CPU budget {detected_cpu_budget}"
        )
    return resolved


def _resolve_reducer_workers(value: AutoInt) -> int:
    if value == "auto":
        return 0
    resolved = int(value)
    if resolved != 0:
        raise ValueError(
            "reducer_workers>0 is deferred until a real reducer stage exists; "
            "use reducer_workers=0 or 'auto'"
        )
    return 0


def _resolve_auto_bool(value: AutoBool, *, default: bool, name: str) -> bool:
    if value == "auto":
        return bool(default)
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be auto, true, or false")


def _resolve_teacher_device(value: str) -> str:
    if value not in SUPPORTED_TEACHER_DEVICES:
        raise ValueError(f"teacher_device must be one of {SUPPORTED_TEACHER_DEVICES}")
    if value != "auto":
        if value in {"cuda", "mps"} and not _device_available(value):
            raise ValueError(f"teacher_device={value!r} is not available")
        return value
    if _device_available("cuda"):
        return "cuda"
    if _device_available("mps"):
        return "mps"
    return "cpu"


def _device_available(device: str) -> bool:
    try:
        import torch
    except ImportError:
        return False
    if device == "cuda":
        return bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    if device == "mps":
        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None)
        return bool(mps is not None and mps.is_available())
    return device == "cpu"


def _gpu_name(device: str) -> str | None:
    if device == "cuda":
        try:
            import torch

            return str(torch.cuda.get_device_name(0))
        except Exception:
            return None
    if device == "mps":
        return "mps"
    return None


def _gpu_reduction_mode(device: str) -> str:
    if device in {"cuda", "mps"}:
        return "full_logits_host_transfer_fallback"
    return "not_applicable"


def _validate_depth(value: int, *, name: str) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be > 0")
