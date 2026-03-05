"""Typed errors used across the CAD pipeline."""

from __future__ import annotations


class CADCLIError(Exception):
    """Base error with a machine-readable failure category."""

    category = "unknown"

    def __init__(self, message: str, *, category: str | None = None):
        super().__init__(message)
        if category:
            self.category = category


class SchemaInvalidError(CADCLIError):
    category = "schema_invalid"


class TopologyInvalidError(CADCLIError):
    category = "topology_invalid"


class AlignmentLowError(CADCLIError):
    category = "alignment_low"


class ExportFailedError(CADCLIError):
    category = "export_failed"
