"""Student model interfaces for QRWKV-XLA."""

from qrwkv_xla.students.backend import StudentBackend
from qrwkv_xla.students.base import StudentModel, StudentOutput
from qrwkv_xla.students.config_selection import (
    SelectedStudentConfig,
    qrwkv_student_config_from_vocab_contract,
)
from qrwkv_xla.students.current_backend import (
    CurrentQRWKVStudentBackend,
    create_current_qrwkv_student_backend,
)
from qrwkv_xla.students.factory import create_student
from qrwkv_xla.students.lm_head import (
    apply_lm_head,
    apply_tied_lm_head,
    init_lm_head_params,
)
from qrwkv_xla.students.pallas_wkv import (
    PallasWKVParityCase,
    PallasWKVSequenceParityCase,
    pallas_available,
    pallas_wkv_sequence_parity_cases,
    pallas_wkv_sequence_update_fused_or_scan,
    pallas_wkv_sequence_update_repeated,
    pallas_wkv_shape_dtype_parity_cases,
    pallas_wkv_update,
    reference_wkv_sequence_update,
    reference_wkv_update,
    run_minimal_pallas_wkv_probe,
    run_pallas_wkv_fused_sequence_parity_matrix,
    run_pallas_wkv_parity_probe,
    run_pallas_wkv_sequence_parity_matrix,
    run_pallas_wkv_shape_dtype_parity_matrix,
)
from qrwkv_xla.students.registry import (
    CURRENT_QRWKV_ARCHITECTURE_ID,
    StudentBackendSpec,
    available_student_architectures,
    create_student_backend,
    student_backend_spec,
)
from qrwkv_xla.students.rwkv7_qwen_reference import (
    RWKV7QwenReferenceConfig,
    RWKV7QwenReferenceState,
    RWKV7QwenReferenceStudent,
    rwkv7_qwen_reference_group_kv,
    rwkv7_qwen_reference_initial_state,
    rwkv7_qwen_reference_rope,
)
from qrwkv_xla.students.rwkv7_radlads_reference import (
    RWKV7RADLADSReferenceConfig,
    RWKV7RADLADSReferenceStudent,
    rwkv7_radlads_reference_initial_state,
    rwkv7_radlads_reference_layer,
)
from qrwkv_xla.students.rwkv7_reference import (
    RWKV7ReferenceConfig,
    RWKV7ReferenceStudent,
    rwkv7_reference_layer,
)
from qrwkv_xla.students.student_runtime import (
    PallasStudentRuntime,
    ReferenceJaxStudentRuntime,
    StudentRuntime,
    create_student_runtime,
)
from qrwkv_xla.students.tiny import TinyStudent, TinyStudentConfig
from qrwkv_xla.students.tiny_debug_backend import (
    TINY_DEBUG_ARCHITECTURE_ID,
    TinyDebugState,
    TinyDebugStudentBackend,
)
from qrwkv_xla.students.wkv_runtime import (
    PallasRuntimeUnavailableError,
    WKVRuntime,
    build_pallas_runtime_probe,
    normalize_wkv_runtime,
)

__all__ = [
    "RWKV7ReferenceConfig",
    "RWKV7ReferenceStudent",
    "RWKV7RADLADSReferenceConfig",
    "RWKV7RADLADSReferenceStudent",
    "RWKV7QwenReferenceConfig",
    "RWKV7QwenReferenceState",
    "RWKV7QwenReferenceStudent",
    "PallasRuntimeUnavailableError",
    "PallasWKVParityCase",
    "PallasWKVSequenceParityCase",
    "CURRENT_QRWKV_ARCHITECTURE_ID",
    "StudentModel",
    "StudentBackend",
    "StudentBackendSpec",
    "StudentOutput",
    "StudentRuntime",
    "SelectedStudentConfig",
    "ReferenceJaxStudentRuntime",
    "PallasStudentRuntime",
    "TinyStudent",
    "TinyStudentConfig",
    "TINY_DEBUG_ARCHITECTURE_ID",
    "TinyDebugState",
    "TinyDebugStudentBackend",
    "WKVRuntime",
    "apply_lm_head",
    "apply_tied_lm_head",
    "build_pallas_runtime_probe",
    "available_student_architectures",
    "CurrentQRWKVStudentBackend",
    "create_student",
    "create_student_backend",
    "create_current_qrwkv_student_backend",
    "qrwkv_student_config_from_vocab_contract",
    "create_student_runtime",
    "init_lm_head_params",
    "normalize_wkv_runtime",
    "pallas_available",
    "pallas_wkv_sequence_update_fused_or_scan",
    "pallas_wkv_sequence_parity_cases",
    "pallas_wkv_sequence_update_repeated",
    "pallas_wkv_shape_dtype_parity_cases",
    "pallas_wkv_update",
    "reference_wkv_sequence_update",
    "reference_wkv_update",
    "rwkv7_radlads_reference_initial_state",
    "rwkv7_radlads_reference_layer",
    "rwkv7_qwen_reference_group_kv",
    "rwkv7_qwen_reference_initial_state",
    "rwkv7_qwen_reference_rope",
    "rwkv7_reference_layer",
    "student_backend_spec",
    "run_minimal_pallas_wkv_probe",
    "run_pallas_wkv_fused_sequence_parity_matrix",
    "run_pallas_wkv_parity_probe",
    "run_pallas_wkv_sequence_parity_matrix",
    "run_pallas_wkv_shape_dtype_parity_matrix",
]
