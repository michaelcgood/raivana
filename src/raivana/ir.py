"""Raivana Intermediate Representation (IR).

Every translation engine (rules-based NLP, Claude, or hybrid) produces this IR, and
every output generator (XACML, NGAC, Cedar, Rego) consumes it. Keeping the language
model's job limited to producing a small, validated IR -- rather than raw policy
code -- is what makes the generated policies deterministic, well-formed, and cheap.

Maps to US 10,606,957 B1: a "candidate rule" populated with "candidate attributes"
(claims 1, 8, 15) before being translated into logical access control policy code.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

Category = Literal["subject", "resource", "action", "environment"]
DataType = Literal["string", "integer", "double", "boolean"]
Scalar = Union[bool, int, float, str]


class Effect(str, Enum):
    PERMIT = "Permit"
    DENY = "Deny"

    def opposite(self) -> Effect:
        return Effect.DENY if self is Effect.PERMIT else Effect.PERMIT


class Operator(str, Enum):
    EQ = "eq"
    NEQ = "neq"
    IN = "in"
    NOT_IN = "not_in"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"

    @property
    def symbol(self) -> str:
        return {
            "eq": "==", "neq": "!=", "in": "in", "not_in": "not in",
            "gt": ">", "gte": ">=", "lt": "<", "lte": "<=",
        }[self.value]

    def negate(self) -> Operator:
        return {
            Operator.EQ: Operator.NEQ, Operator.NEQ: Operator.EQ,
            Operator.IN: Operator.NOT_IN, Operator.NOT_IN: Operator.IN,
            Operator.GT: Operator.LTE, Operator.LTE: Operator.GT,
            Operator.LT: Operator.GTE, Operator.GTE: Operator.LT,
        }[self]

    @property
    def is_ordering(self) -> bool:
        return self in (Operator.GT, Operator.GTE, Operator.LT, Operator.LTE)


class CombiningAlgorithm(str, Enum):
    DENY_OVERRIDES = "deny-overrides"
    PERMIT_OVERRIDES = "permit-overrides"
    FIRST_APPLICABLE = "first-applicable"
    DENY_UNLESS_PERMIT = "deny-unless-permit"
    PERMIT_UNLESS_DENY = "permit-unless-deny"


class AttributeRef(BaseModel):
    """A reference to an attribute, e.g. subject.department."""

    model_config = ConfigDict(frozen=True)

    category: Category
    id: str
    datatype: DataType = "string"

    @property
    def path(self) -> str:
        return f"{self.category}.{self.id}"

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.path


class Condition(BaseModel):
    """`attribute <operator> value` -- value may be a literal, a list, or another attribute."""

    attribute: AttributeRef
    operator: Operator
    value: Union[AttributeRef, list[Scalar], Scalar]

    @model_validator(mode="after")
    def _normalize_value_shape(self) -> Condition:
        op = self.operator
        if op in (Operator.IN, Operator.NOT_IN) and not isinstance(self.value, (list, AttributeRef)):
            self.value = [self.value]
        if op not in (Operator.IN, Operator.NOT_IN) and isinstance(self.value, list):
            if len(self.value) != 1:
                raise ValueError(f"operator {op.value} does not accept a list of values")
            self.value = self.value[0]
        return self

    @property
    def is_reference(self) -> bool:
        return isinstance(self.value, AttributeRef)

    @property
    def values(self) -> list[Scalar]:
        if isinstance(self.value, AttributeRef):
            return []
        return list(self.value) if isinstance(self.value, list) else [self.value]

    def negated(self) -> Condition:
        return Condition(attribute=self.attribute, operator=self.operator.negate(), value=self.value)

    def describe(self) -> str:
        if isinstance(self.value, AttributeRef):
            rhs = self.value.path
        elif isinstance(self.value, list):
            rhs = "[" + ", ".join(json.dumps(v) for v in self.value) + "]"
        else:
            rhs = json.dumps(self.value)
        return f"{self.attribute.path} {self.operator.symbol} {rhs}"


class Rule(BaseModel):
    """A single access control rule. All condition lists are AND-ed; `actions` is OR-ed."""

    id: str
    effect: Effect
    description: str = ""
    subject: list[Condition] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list, description="Empty means any action.")
    resource: list[Condition] = Field(default_factory=list)
    environment: list[Condition] = Field(default_factory=list)
    source_text: str = ""
    confidence: float = 1.0
    template: str | None = Field(default=None, description="Sample rule (policy set entry) used.")
    engine: str = "manual"

    def all_conditions(self) -> list[Condition]:
        return [*self.subject, *self.resource, *self.environment]


class Policy(BaseModel):
    """A policy: an ordered set of rules plus a rule-combining algorithm."""

    id: str
    description: str = ""
    combining: CombiningAlgorithm = CombiningAlgorithm.DENY_OVERRIDES
    default_effect: Effect | None = Field(
        default=None, description="Decision when no rule applies (None = NotApplicable).")
    rules: list[Rule] = Field(default_factory=list)
    language: str = "en"
    policy_type: str = "general"
    source_text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json", exclude={"metadata"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def attributes(self) -> list[AttributeRef]:
        seen: dict[str, AttributeRef] = {}
        for rule in self.rules:
            for cond in rule.all_conditions():
                seen.setdefault(cond.attribute.path, cond.attribute)
                if isinstance(cond.value, AttributeRef):
                    seen.setdefault(cond.value.path, cond.value)
        return list(seen.values())

    def actions(self) -> list[str]:
        out: list[str] = []
        for rule in self.rules:
            for a in rule.actions:
                if a not in out:
                    out.append(a)
        return out

    @classmethod
    def load(cls, path: str) -> Policy:
        with open(path, encoding="utf-8") as fh:
            return cls.model_validate_json(fh.read())

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.model_dump_json(indent=2))


class AccessRequest(BaseModel):
    """An access request to evaluate against a policy (used by the policy tester)."""

    subject: dict[str, Any] = Field(default_factory=dict)
    action: str
    resource: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)

    def get(self, ref: AttributeRef) -> Any:
        if ref.category == "action":
            return self.action if ref.id in ("id", "action") else None
        bag = getattr(self, ref.category)
        return bag.get(ref.id)
