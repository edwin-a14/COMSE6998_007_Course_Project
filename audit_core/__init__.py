"""Shared audit interface and general auditors, independent of any benchmark.

See aftraj/README.md for the interface specification.
"""

from audit_core.composite import AllowAllAuditor, CompositeAuditor
from audit_core.interface import (
    AuditContext,
    Auditor,
    ToolCall,
    ToolSpec,
    Turn,
    Verdict,
)

__all__ = [
    "AllowAllAuditor",
    "AuditContext",
    "Auditor",
    "CompositeAuditor",
    "ToolCall",
    "ToolSpec",
    "Turn",
    "Verdict",
]
