"""Well-formedness checking (US 10,606,957 B1, FIG. 3 step 308; FIG. 6 step 610).

Checks that a populated rule is a complete, type-correct, logically meaningful
statement before any policy code is generated from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .analysis import solve
from .dictionary import AttributeDictionary
from .ir import AttributeRef, Condition, Effect, Operator, Policy, Rule


@dataclass
class Issue:
    severity: str  # error | warning
    rule: str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _check_condition(rule: Rule, c: Condition, dictionary: AttributeDictionary | None) -> list[Issue]:
    out: list[Issue] = []
    dt = c.attribute.datatype
    if c.operator.is_ordering:
        if dt not in ("integer", "double"):
            out.append(Issue("error", rule.id, f"'{c.describe()}': ordering comparison on non-numeric "
                                               f"attribute {c.attribute.path} ({dt})."))
        elif not isinstance(c.value, AttributeRef) and not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in c.values):
            out.append(Issue("error", rule.id, f"'{c.describe()}': comparison value must be numeric."))
    if c.operator in (Operator.IN, Operator.NOT_IN) and not isinstance(c.value, AttributeRef) and not c.values:
        out.append(Issue("error", rule.id, f"'{c.describe()}': empty value list."))
    if dt == "boolean" and not isinstance(c.value, AttributeRef) and \
            not all(isinstance(v, bool) for v in c.values):
        out.append(Issue("error", rule.id, f"'{c.describe()}': boolean attribute compared to non-boolean."))
    if dt == "integer" and not isinstance(c.value, AttributeRef) and \
            not all(isinstance(v, int) and not isinstance(v, bool) for v in c.values):
        out.append(Issue("error", rule.id, f"'{c.describe()}': integer attribute compared to non-integer."))
    if isinstance(c.value, AttributeRef) and c.value.path == c.attribute.path:
        out.append(Issue("error", rule.id, f"'{c.describe()}': attribute compared to itself."))
    if isinstance(c.value, AttributeRef) and c.operator.is_ordering:
        out.append(Issue("warning", rule.id, f"'{c.describe()}': ordering between attributes is not "
                                             f"supported by every target language."))
    if dictionary is not None:
        for ref in [c.attribute, *( [c.value] if isinstance(c.value, AttributeRef) else [])]:
            attr = dictionary.get(ref.id)
            if attr is None:
                out.append(Issue("warning", rule.id, f"Attribute {ref.path} is not in the attribute dictionary."))
            elif ref.category not in attr.categories and ref.id not in ("id", "owner", "assignee"):
                out.append(Issue("warning", rule.id, f"Attribute {ref.id} is not normally a {ref.category} "
                                                     f"attribute (dictionary: {', '.join(attr.categories)})."))
    return out


def check_rule(rule: Rule, dictionary: AttributeDictionary | None = None) -> list[Issue]:
    issues: list[Issue] = []
    if not rule.id:
        issues.append(Issue("error", None, "Rule has no id."))
    for c in rule.all_conditions():
        issues.extend(_check_condition(rule, c, dictionary))
    for cat, conds in (("subject", rule.subject), ("resource", rule.resource), ("environment", rule.environment)):
        for c in conds:
            if c.attribute.category != cat:
                issues.append(Issue("error", rule.id, f"'{c.describe()}' is filed under {cat} conditions."))
    if not any(i.severity == "error" for i in issues):
        ok, _ = solve(rule.all_conditions())
        if not ok:
            issues.append(Issue("error", rule.id, "Conditions are contradictory; the rule can never apply."))
    if not rule.all_conditions() and not rule.actions and rule.effect == Effect.DENY:
        issues.append(Issue("warning", rule.id, "Rule denies every action on every resource for everyone."))
    if not rule.subject and not rule.resource and not rule.environment and rule.effect == Effect.PERMIT:
        issues.append(Issue("warning", rule.id, "Rule permits everyone on every resource."))
    if dictionary is not None:
        for a in rule.actions:
            if a not in dictionary.actions:
                issues.append(Issue("warning", rule.id, f"Action '{a}' is not in the attribute dictionary."))
    return issues


def check_policy(policy: Policy, dictionary: AttributeDictionary | None = None) -> list[Issue]:
    issues: list[Issue] = []
    if not policy.id:
        issues.append(Issue("error", None, "Policy has no id."))
    if not policy.rules:
        issues.append(Issue("warning", None, "Policy contains no rules."))
    seen: set[str] = set()
    for r in policy.rules:
        if r.id in seen:
            issues.append(Issue("error", r.id, f"Duplicate rule id {r.id}."))
        seen.add(r.id)
        issues.extend(check_rule(r, dictionary))
    return issues


def is_well_formed(rule: Rule, dictionary: AttributeDictionary | None = None) -> bool:
    return not any(i.severity == "error" for i in check_rule(rule, dictionary))
