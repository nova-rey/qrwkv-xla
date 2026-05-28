from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from qrwkv_xla.kernels.wkv7_fixtures import wkv7_reference_full_scan
from qrwkv_xla.students.pallas_wkv import pallas_available

SUPPORTED_CANDIDATES = ("reference", "pallas")


@dataclass(frozen=True)
class UnsupportedCandidate(Exception):
    candidate: str
    reason: str


def run_wkv7_candidate(
    candidate: str, inputs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    if candidate == "reference":
        return wkv7_reference_full_scan(inputs)
    if candidate == "pallas":
        return wkv7_pallas_fixture_full_scan(inputs)
    raise UnsupportedCandidate(candidate=candidate, reason="unknown candidate")


def wkv7_pallas_fixture_full_scan(
    inputs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    available, reason = pallas_available()
    if not available:
        raise UnsupportedCandidate(
            candidate="pallas",
            reason=reason or "missing_pallas_dependency_or_backend",
        )

    state = jnp.asarray(inputs["initial_state"], dtype=jnp.float32)
    batch_size, sequence_length = inputs["r"].shape[:2]
    mask = inputs.get("attention_mask")
    masks = (
        jnp.ones((batch_size, sequence_length), dtype=jnp.float32)
        if mask is None
        else jnp.asarray(mask, dtype=jnp.float32)
    )
    xs = (
        jnp.swapaxes(jnp.asarray(inputs["r"], dtype=jnp.float32), 0, 1),
        jnp.swapaxes(jnp.asarray(inputs["w"], dtype=jnp.float32), 0, 1),
        jnp.swapaxes(jnp.asarray(inputs["k"], dtype=jnp.float32), 0, 1),
        jnp.swapaxes(jnp.asarray(inputs["v"], dtype=jnp.float32), 0, 1),
        jnp.swapaxes(jnp.asarray(inputs["a"], dtype=jnp.float32), 0, 1),
        jnp.swapaxes(jnp.asarray(inputs["b"], dtype=jnp.float32), 0, 1),
        jnp.swapaxes(jnp.asarray(inputs["gate"], dtype=jnp.float32), 0, 1),
        jnp.swapaxes(masks, 0, 1),
    )
    next_state, outputs = jax.lax.scan(_pallas_wkv7_fixture_step, state, xs)
    outputs = jnp.swapaxes(outputs, 0, 1)
    outputs.block_until_ready()
    return {
        "output": np.asarray(jax.device_get(outputs), dtype=np.float32),
        "next_state": np.asarray(jax.device_get(next_state), dtype=np.float32),
    }


def _pallas_wkv7_fixture_step(
    carry: jax.Array,
    item: tuple[jax.Array, ...],
) -> tuple[jax.Array, jax.Array]:
    import jax.experimental.pallas as pl

    token_r, token_w, token_k, token_v, token_a, token_b, token_gate, token_mask = item
    _validate_fixture_step_shapes(
        carry,
        token_r,
        token_w,
        token_k,
        token_v,
        token_a,
        token_b,
        token_gate,
        token_mask,
    )

    def kernel(
        state_ref,
        r_ref,
        w_ref,
        k_ref,
        v_ref,
        a_ref,
        b_ref,
        gate_ref,
        mask_ref,
        next_state_ref,
        output_ref,
    ):
        state_block = state_ref[:]
        token_r_block = r_ref[:]
        token_w_block = w_ref[:]
        token_k_block = k_ref[:]
        token_v_block = v_ref[:]
        token_a_block = a_ref[:]
        token_b_block = b_ref[:]
        token_gate_block = gate_ref[:]
        token_mask_block = mask_ref[:]
        token_mask_state = token_mask_block.reshape(token_mask_block.shape[0], 1, 1)
        masked_v = token_v_block * token_mask_state
        kk = token_k_block / jnp.maximum(
            jnp.linalg.norm(token_k_block, axis=-1, keepdims=True),
            jnp.asarray(1e-6, dtype=token_k_block.dtype),
        )
        log_w = -jnp.exp(jnp.asarray(-0.5, dtype=token_w_block.dtype)) * jax.nn.sigmoid(
            token_w_block
        )
        decay = jnp.exp(log_w)
        vk = jnp.einsum("bhi,bhj->bhij", masked_v, token_k_block)
        ab = jnp.einsum(
            "bhi,bhj->bhij",
            -kk,
            kk * token_a_block + token_b_block,
        )
        next_state = state_block * decay[:, :, None, :] + state_block @ ab + vk
        output = jnp.einsum("bhij,bhj->bhi", next_state, token_r_block) * (
            token_gate_block
        )
        output = output * token_mask_state
        next_state_ref[:] = next_state
        output_ref[:] = output

    run_kernel = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(carry.shape, carry.dtype),
            jax.ShapeDtypeStruct(token_r.shape, token_r.dtype),
        ),
        interpret=True,
        name="pallas_wkv7_fixture_family_step",
    )
    next_state, output = run_kernel(
        carry,
        token_r,
        token_w,
        token_k,
        token_v,
        token_a,
        token_b,
        token_gate,
        token_mask,
    )
    return next_state, output


def _validate_fixture_step_shapes(
    state: jax.Array,
    token_r: jax.Array,
    token_w: jax.Array,
    token_k: jax.Array,
    token_v: jax.Array,
    token_a: jax.Array,
    token_b: jax.Array,
    token_gate: jax.Array,
    token_mask: jax.Array,
) -> None:
    if state.ndim != 4:
        raise ValueError(f"state must have shape [B,H,D,D], got {state.shape}")
    expected_vector_shape = state.shape[:3]
    for name, value in {
        "r": token_r,
        "w": token_w,
        "k": token_k,
        "v": token_v,
        "a": token_a,
        "b": token_b,
        "gate": token_gate,
    }.items():
        if value.shape != expected_vector_shape:
            raise ValueError(
                f"{name} must have shape {expected_vector_shape}, got {value.shape}"
            )
    if token_mask.shape != state.shape[:1]:
        raise ValueError(
            f"mask must have shape {state.shape[:1]}, got {token_mask.shape}"
        )
