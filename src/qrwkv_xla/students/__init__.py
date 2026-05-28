"""Student model interfaces for QRWKV-XLA."""

from qrwkv_xla.students.base import StudentModel, StudentOutput
from qrwkv_xla.students.factory import create_student
from qrwkv_xla.students.lm_head import (
    apply_lm_head,
    apply_tied_lm_head,
    init_lm_head_params,
)
from qrwkv_xla.students.pallas_wkv import (
    PallasWKVParityCase,
    pallas_available,
    pallas_wkv_update,
    pallas_wkv_shape_dtype_parity_cases,
    reference_wkv_update,
    run_minimal_pallas_wkv_probe,
    run_pallas_wkv_shape_dtype_parity_matrix,
    run_pallas_wkv_parity_probe,
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
from qrwkv_xla.students.tiny import TinyStudent, TinyStudentConfig
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
    "StudentModel",
    "StudentOutput",
    "TinyStudent",
    "TinyStudentConfig",
    "WKVRuntime",
    "apply_lm_head",
    "apply_tied_lm_head",
    "build_pallas_runtime_probe",
    "create_student",
    "init_lm_head_params",
    "normalize_wkv_runtime",
    "pallas_available",
    "pallas_wkv_update",
    "pallas_wkv_shape_dtype_parity_cases",
    "reference_wkv_update",
    "rwkv7_radlads_reference_initial_state",
    "rwkv7_radlads_reference_layer",
    "rwkv7_qwen_reference_group_kv",
    "rwkv7_qwen_reference_initial_state",
    "rwkv7_qwen_reference_rope",
    "rwkv7_reference_layer",
    "run_minimal_pallas_wkv_probe",
    "run_pallas_wkv_shape_dtype_parity_matrix",
    "run_pallas_wkv_parity_probe",
]
