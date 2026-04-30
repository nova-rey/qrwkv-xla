from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JaxRuntimeInfo:
    jax_available: bool
    jax_version: str | None
    default_backend: str | None
    devices: tuple[str, ...]
    platforms: tuple[str, ...]
    has_cpu: bool
    has_gpu: bool
    has_tpu: bool


def get_jax_runtime_info() -> JaxRuntimeInfo:
    try:
        import jax  # type: ignore
    except ImportError:
        return JaxRuntimeInfo(
            jax_available=False,
            jax_version=None,
            default_backend=None,
            devices=(),
            platforms=(),
            has_cpu=False,
            has_gpu=False,
            has_tpu=False,
        )

    devices = tuple(_format_device(device) for device in jax.devices())
    platforms = tuple(
        sorted(
            {str(getattr(device, "platform", "unknown")) for device in jax.devices()}
        )
    )
    return JaxRuntimeInfo(
        jax_available=True,
        jax_version=str(jax.__version__),
        default_backend=str(jax.default_backend()),
        devices=devices,
        platforms=platforms,
        has_cpu="cpu" in platforms,
        has_gpu="gpu" in platforms,
        has_tpu="tpu" in platforms,
    )


def format_jax_runtime_info(info: JaxRuntimeInfo) -> str:
    lines = [f"jax_available: {info.jax_available}"]
    if not info.jax_available:
        return "\n".join(lines)

    lines.extend(
        [
            f"jax_version: {info.jax_version}",
            f"default_backend: {info.default_backend}",
            f"platforms: {', '.join(info.platforms) if info.platforms else '(none)'}",
            f"has_cpu: {info.has_cpu}",
            f"has_gpu: {info.has_gpu}",
            f"has_tpu: {info.has_tpu}",
            "devices:",
        ]
    )
    if info.devices:
        lines.extend(f"- {device}" for device in info.devices)
    else:
        lines.append("- none")
    return "\n".join(lines)


def _format_device(device: object) -> str:
    platform = str(getattr(device, "platform", "unknown"))
    device_id = getattr(device, "id", "?")
    kind = str(getattr(device, "device_kind", type(device).__name__))
    process_index = getattr(device, "process_index", None)
    if callable(process_index):
        process_index = process_index()
    suffix = "" if process_index is None else f", process_index={process_index}"
    return f"id={device_id}, platform={platform}, kind={kind}{suffix}"
