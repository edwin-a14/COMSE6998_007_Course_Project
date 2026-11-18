"""General auditors. Each one depends only on `audit_core.interface`."""

from audit_core.auditors.rules import GENERIC_CONFIG, DomainConfig, RulesAuditor

__all__ = ["DomainConfig", "GENERIC_CONFIG", "RulesAuditor"]
