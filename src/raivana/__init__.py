"""Raivana: translate natural-language access control policy into XACML, NGAC, Cedar and Rego.

Reference implementation of US Patent 10,606,957 B1, "Method and system for translating
natural language policy to logical access control policy" (Michael Good & Benjamin Coffinberger).

    >>> import raivana
    >>> result = raivana.translate("Only doctors can prescribe medication.", formats=["cedar"])
    >>> print(result.artifacts["cedar"].content)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .analysis import analyze, evaluate
from .composer import PolicyComposer, TranslationResult
from .dictionary import AttributeDictionary
from .explain import describe_policy, describe_rule
from .generators import FORMATS, generate
from .ir import AccessRequest, AttributeRef, CombiningAlgorithm, Condition, Effect, Operator, Policy, Rule
from .policysets import PolicySets

__version__ = "0.1.0"


def translate(text: str, formats: Iterable[str] = ("json",), engine: str = "auto",
              model: str | None = None, policy_id: str | None = None, **kwargs: Any) -> TranslationResult:
    """One-shot convenience wrapper around :class:`PolicyComposer`."""
    composer = PolicyComposer(engine=engine, model=model, **kwargs)
    return composer.translate(text, policy_id=policy_id, formats=formats)


__all__ = [
    "AccessRequest", "AttributeDictionary", "AttributeRef", "CombiningAlgorithm", "Condition", "Effect",
    "FORMATS", "Operator", "Policy", "PolicyComposer", "PolicySets", "Rule", "TranslationResult",
    "analyze", "describe_policy", "describe_rule", "evaluate", "generate", "translate", "__version__",
]
