"""Streaming dataset helpers for local/offline QRWKV-XLA dry-runs."""

from qrwkv_xla.data.streaming import (
    BOUNDARY_POLICY,
    PHASE,
    STREAMING_DATASET_CREATED_BY,
    STREAMING_DATASET_FORMAT,
    STREAMING_DATASET_SCHEMA_VERSION,
    StreamingBatch,
    StreamingCorpusInfo,
    StreamingCursor,
    StreamingDataset,
    StreamingDatasetManifest,
    StreamingSourceInfo,
    StreamingTokenShardManifest,
    build_streaming_dataset_from_tokenized_corpus,
    read_streaming_dataset_manifest,
    validate_streaming_dataset_manifest,
)
from qrwkv_xla.data.tiny_dataset import (
    TinyTextExample,
    batch_tiny_text_examples,
)
from qrwkv_xla.data.tiny_dataset_pipeline import (
    TinyDatasetPipelineResult,
    run_tiny_dataset_pipeline_smoke,
)

__all__ = [
    "BOUNDARY_POLICY",
    "PHASE",
    "STREAMING_DATASET_CREATED_BY",
    "STREAMING_DATASET_FORMAT",
    "STREAMING_DATASET_SCHEMA_VERSION",
    "StreamingBatch",
    "StreamingCorpusInfo",
    "StreamingCursor",
    "StreamingDataset",
    "StreamingDatasetManifest",
    "StreamingSourceInfo",
    "StreamingTokenShardManifest",
    "TinyDatasetPipelineResult",
    "TinyTextExample",
    "batch_tiny_text_examples",
    "build_streaming_dataset_from_tokenized_corpus",
    "read_streaming_dataset_manifest",
    "validate_streaming_dataset_manifest",
    "run_tiny_dataset_pipeline_smoke",
]
