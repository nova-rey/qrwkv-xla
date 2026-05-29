from __future__ import annotations

from dataclasses import dataclass, replace

from qrwkv_xla.contracts import VocabContract
from qrwkv_xla.students.rwkv7_qwen_reference import RWKV7QwenReferenceConfig
from qrwkv_xla.students.wkv_runtime import WKVRuntime


@dataclass(frozen=True)
class SelectedStudentConfig:
    architecture: str
    config: RWKV7QwenReferenceConfig
    vocab_contract: VocabContract
    runtime: WKVRuntime


def qrwkv_student_config_from_vocab_contract(
    contract: VocabContract,
    *,
    base_config: RWKV7QwenReferenceConfig | None = None,
    runtime: str | WKVRuntime | None = None,
) -> SelectedStudentConfig:
    source_config = base_config or RWKV7QwenReferenceConfig(
        vocab_size=contract.vocab_size,
        hidden_size=4,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        emit_logits=True,
    )
    selected_runtime = (
        source_config.effective_wkv_runtime if runtime is None else runtime
    )
    selected_config = replace(
        source_config,
        vocab_size=contract.vocab_size,
        wkv_runtime=selected_runtime,
    )
    return SelectedStudentConfig(
        architecture="rwkv7_qwen_reference",
        config=selected_config,
        vocab_contract=contract,
        runtime=selected_config.effective_wkv_runtime,
    )
