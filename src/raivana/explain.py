"""Plain-language summaries of generated rules (US 10,606,957 B1, FIG. 7 step 710).

Back-translating the IR into English lets a reviewer confirm the machine's
understanding before the policy is exported or deployed (FIG. 3, step 312).
"""

from __future__ import annotations

from .ir import AttributeRef, Condition, Operator, Policy, Rule

_DAYS = {"mon": "Monday", "tue": "Tuesday", "wed": "Wednesday", "thu": "Thursday", "fri": "Friday",
         "sat": "Saturday", "sun": "Sunday"}


def _fmt(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v).replace("_", " ")


def _hour(h: object) -> str:
    try:
        x = int(h)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return str(h)
    return f"{x:02d}:00"


def describe_condition(c: Condition) -> str:
    name = c.attribute.id.replace("_", " ")
    if isinstance(c.value, AttributeRef):
        other = f"the {c.value.category}'s {c.value.id.replace('_', ' ')}"
        if c.attribute.id == "owner" and c.value.id == "id":
            return "the resource belongs to them" if c.operator in (Operator.EQ, Operator.IN) \
                else "the resource does not belong to them"
        if c.attribute.id == "assignee" and c.value.id == "id":
            return "the resource is assigned to them"
        rel = "matches" if c.operator in (Operator.EQ, Operator.IN) else "differs from"
        return f"the {c.attribute.category}'s {name} {rel} {other}"
    if c.attribute.id == "hour":
        if c.operator in (Operator.NOT_IN, Operator.IN) and len(c.values) > 1:
            lo, hi = min(c.values), max(c.values)  # type: ignore[type-var]
            span = f"{_hour(lo)}-{_hour(int(hi) + 1)}"  # type: ignore[arg-type]
            return f"the time is {'outside' if c.operator == Operator.NOT_IN else 'within'} {span}"
        word = {Operator.GTE: "at or after", Operator.GT: "after", Operator.LT: "before",
                Operator.LTE: "at or before", Operator.EQ: "at", Operator.NEQ: "not at"}.get(c.operator, "")
        return f"the time is {word} {_hour(c.values[0])}"
    if c.attribute.id == "day_of_week":
        days = ", ".join(_DAYS.get(str(v), str(v)) for v in c.values)
        return f"the day is {'not ' if c.operator in (Operator.NEQ, Operator.NOT_IN) else ''}{days}"
    if c.attribute.datatype == "boolean":
        truth = c.values[0] if c.values else True
        positive = (c.operator == Operator.EQ) == bool(truth)
        return f"{name} is {'satisfied' if positive else 'not satisfied'}"
    vals = c.values
    joined = " or ".join(_fmt(v) for v in vals)
    phrase = {
        Operator.EQ: f"{name} is {joined}", Operator.IN: f"{name} is {joined}",
        Operator.NEQ: f"{name} is not {joined}", Operator.NOT_IN: f"{name} is not {joined}",
        Operator.GT: f"{name} is greater than {joined}", Operator.GTE: f"{name} is at least {joined}",
        Operator.LT: f"{name} is less than {joined}", Operator.LTE: f"{name} is at most {joined}",
    }[c.operator]
    return phrase


def describe_rule(rule: Rule) -> str:
    verb = "ALLOW" if rule.effect.value == "Permit" else "DENY"
    who = " and ".join(describe_condition(c) for c in rule.subject) or "anyone"
    if rule.subject:
        who = f"users whose {who}"
    acts = ", ".join(a.replace("_", " ") for a in rule.actions) if rule.actions else "any action"
    what = " and ".join(describe_condition(c) for c in rule.resource)
    what = f"resources where {what}" if what else "any resource"
    when = " and ".join(describe_condition(c) for c in rule.environment)
    out = f"{verb} {who} to perform {acts} on {what}"
    if when:
        out += f", when {when}"
    return out + "."


def describe_policy(policy: Policy) -> str:
    lines = [f"Policy '{policy.id}' ({policy.combining.value}"
             + (f", default {policy.default_effect.value}" if policy.default_effect else "") + "):"]
    for r in policy.rules:
        lines.append(f"  [{r.id}] {describe_rule(r)}")
    return "\n".join(lines)
