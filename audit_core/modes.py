"""Build comparable model-based auditors without benchmark dependencies."""

from audit_core.composite import CompositeAuditor
from audit_core.interface import Auditor
from audit_core.auditors import GENERIC_CONFIG, DomainConfig, RulesAuditor, SelectiveSemanticAuditor, SemanticAuditor
from audit_core.model import ModelBackend


MODEL_MODES = ("general-hybrid", "selective-semantic", "continuous-semantic")


def build_model_auditor(
    mode: str, *, model: str, backend: ModelBackend | None = None,
    rules_config: DomainConfig = GENERIC_CONFIG, max_tokens: int = 512, timeout: float = 30,
) -> Auditor:
    """Share model, prompt and selection policy across the experimental controls.

    AFTraj uses generic rules; retail callers must supply the retail config.
    Construction never calls the model. This does not install runtime hooks.
    """
    if mode not in MODEL_MODES:
        raise ValueError(f"Unknown model auditor mode: {mode}")
    semantic = SemanticAuditor(model, backend=backend, max_tokens=max_tokens, timeout=timeout)
    if mode == "continuous-semantic":
        return semantic
    selective = SelectiveSemanticAuditor(semantic)
    if mode == "selective-semantic":
        return selective
    return CompositeAuditor(RulesAuditor(rules_config), selective, name="general-hybrid")
