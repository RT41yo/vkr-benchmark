"""Diagnostic helpers for Stage-5 benchmark investigations."""

from vkr_benchmark.diagnostics.rrc_boundary_precision import (
    RRCBoundaryPrecisionAnalysis,
    RRCLostInterval,
    analyze_rrc_boundary_precision,
)

from vkr_benchmark.diagnostics.hard_failures import (
    HardFailureRecord,
    classify_hard_failure,
    load_hard_failure_records,
    summarize_hard_failure_diagnostics,
)

__all__ = [
    "HardFailureRecord",
    "classify_hard_failure",
    "load_hard_failure_records",
    "summarize_hard_failure_diagnostics",
    "RRCBoundaryPrecisionAnalysis",
    "RRCLostInterval",
    "analyze_rrc_boundary_precision",
]
