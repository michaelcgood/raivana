"""Policy decision point (tester) and static analysis (conflicts, redundancy, gaps).

The PDP evaluates the IR directly with the same semantics the generators emit:
  * eq / in / ordering on a missing attribute  -> false
  * neq / not_in on a missing attribute        -> true
  * multi-valued request attributes (lists) match if any value matches.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from .ir import AccessRequest, AttributeRef, CombiningAlgorithm, Condition, Effect, Operator, Policy, Rule

# =========================================================================== PDP


@dataclass
class ConditionTrace:
    condition: str
    actual: Any
    result: bool


@dataclass
class RuleTrace:
    rule_id: str
    effect: str
    applicable: bool
    action_match: bool
    conditions: list[ConditionTrace] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"rule": self.rule_id, "effect": self.effect, "applicable": self.applicable,
                "action_match": self.action_match,
                "conditions": [c.__dict__ for c in self.conditions]}


@dataclass
class Decision:
    decision: str  # Permit | Deny | NotApplicable
    determining_rules: list[str]
    trace: list[RuleTrace]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"decision": self.decision, "determining_rules": self.determining_rules,
                "reason": self.reason, "trace": [t.to_dict() for t in self.trace]}


def _coerce(v: Any, datatype: str) -> Any:
    if v is None:
        return None
    try:
        if datatype == "integer":
            return int(v) if not isinstance(v, bool) else int(v)
        if datatype == "double":
            return float(v)
        if datatype == "boolean":
            if isinstance(v, str):
                return v.strip().lower() in ("true", "yes", "1", "on")
            return bool(v)
    except (TypeError, ValueError):
        return v
    return str(v) if not isinstance(v, str) else v


def _values(request: AccessRequest, ref: AttributeRef) -> list[Any]:
    raw = request.get(ref)
    if raw is None:
        return []
    items = raw if isinstance(raw, (list, tuple, set)) else [raw]
    return [_coerce(x, ref.datatype) for x in items]


def eval_condition(cond: Condition, request: AccessRequest) -> tuple[bool, Any]:
    actual = _values(request, cond.attribute)
    op = cond.operator
    if isinstance(cond.value, AttributeRef):
        other = _values(request, cond.value)
        shared = bool(set(map(str, actual)) & set(map(str, other)))
        return (shared if op in (Operator.EQ, Operator.IN) else not shared), actual
    expected = [_coerce(v, cond.attribute.datatype) for v in cond.values]
    if op == Operator.EQ or op == Operator.IN:
        return any(a in expected for a in actual), actual
    if op == Operator.NEQ or op == Operator.NOT_IN:
        return not any(a in expected for a in actual), actual
    target = expected[0]
    try:
        cmp = {
            Operator.GT: lambda a: a > target, Operator.GTE: lambda a: a >= target,
            Operator.LT: lambda a: a < target, Operator.LTE: lambda a: a <= target,
        }[op]
        return any(cmp(a) for a in actual), actual
    except TypeError:
        return False, actual


def rule_applies(rule: Rule, request: AccessRequest) -> RuleTrace:
    action_match = not rule.actions or request.action in rule.actions or "*" in rule.actions
    trace = RuleTrace(rule.id, rule.effect.value, False, action_match)
    ok = action_match
    for cond in rule.all_conditions():
        res, actual = eval_condition(cond, request)
        trace.conditions.append(ConditionTrace(cond.describe(), actual if len(actual) != 1 else actual[0], res))
        ok = ok and res
    trace.applicable = ok
    return trace


def evaluate(policy: Policy, request: AccessRequest | dict[str, Any]) -> Decision:
    """Evaluate an access request against a policy (the built-in policy tester)."""
    if isinstance(request, dict):
        request = AccessRequest.model_validate(request)
    traces = [rule_applies(r, request) for r in policy.rules]
    permits = [t.rule_id for t in traces if t.applicable and t.effect == "Permit"]
    denies = [t.rule_id for t in traces if t.applicable and t.effect == "Deny"]
    alg = policy.combining
    decision, determining, reason = "NotApplicable", [], "No rule applies."
    if alg == CombiningAlgorithm.FIRST_APPLICABLE:
        first = next((t for t in traces if t.applicable), None)
        if first:
            decision, determining = first.effect, [first.rule_id]
            reason = f"First applicable rule is {first.rule_id}."
    elif alg in (CombiningAlgorithm.DENY_OVERRIDES, CombiningAlgorithm.PERMIT_UNLESS_DENY):
        if denies:
            decision, determining, reason = "Deny", denies, "A Deny rule applies and deny overrides."
        elif permits:
            decision, determining, reason = "Permit", permits, "A Permit rule applies and no Deny rule does."
        elif alg == CombiningAlgorithm.PERMIT_UNLESS_DENY:
            decision, reason = "Permit", "No Deny rule applies (permit-unless-deny)."
    elif alg in (CombiningAlgorithm.PERMIT_OVERRIDES, CombiningAlgorithm.DENY_UNLESS_PERMIT):
        if permits:
            decision, determining, reason = "Permit", permits, "A Permit rule applies and permit overrides."
        elif denies:
            decision, determining, reason = "Deny", denies, "A Deny rule applies and no Permit rule does."
        elif alg == CombiningAlgorithm.DENY_UNLESS_PERMIT:
            decision, reason = "Deny", "No Permit rule applies (deny-unless-permit)."
    if decision == "NotApplicable" and policy.default_effect is not None:
        decision = policy.default_effect.value
        reason = f"No rule applies; policy default is {decision}."
    return Decision(decision, determining, traces, reason)


# =========================================================================== constraint solving


@dataclass
class _Domain:
    ref: AttributeRef
    allowed: set[Any] | None = None
    excluded: set[Any] = field(default_factory=set)
    lo: tuple[float, bool] | None = None  # (bound, inclusive)
    hi: tuple[float, bool] | None = None

    def add(self, cond: Condition) -> None:
        op = cond.operator
        vals = [_coerce(v, self.ref.datatype) for v in cond.values]
        if op in (Operator.EQ, Operator.IN):
            s = set(vals)
            self.allowed = s if self.allowed is None else self.allowed & s
        elif op in (Operator.NEQ, Operator.NOT_IN):
            self.excluded |= set(vals)
        elif op in (Operator.GT, Operator.GTE):
            b = (float(vals[0]), op == Operator.GTE)
            if self.lo is None or b[0] > self.lo[0] or (b[0] == self.lo[0] and not b[1]):
                self.lo = b
        elif op in (Operator.LT, Operator.LTE):
            b = (float(vals[0]), op == Operator.LTE)
            if self.hi is None or b[0] < self.hi[0] or (b[0] == self.hi[0] and not b[1]):
                self.hi = b

    def _in_bounds(self, v: Any) -> bool:
        try:
            x = float(v)
        except (TypeError, ValueError):
            return self.lo is None and self.hi is None
        if self.lo and (x < self.lo[0] or (x == self.lo[0] and not self.lo[1])):
            return False
        if self.hi and (x > self.hi[0] or (x == self.hi[0] and not self.hi[1])):
            return False
        return True

    def solve(self) -> tuple[bool, Any]:
        """Return (satisfiable, witness value). Witness None means 'leave attribute absent'."""
        if self.allowed is not None:
            for v in sorted(self.allowed, key=str):
                if v not in self.excluded and self._in_bounds(v):
                    return True, v
            return False, None
        dt = self.ref.datatype
        if dt == "boolean":
            for v in (True, False):
                if v not in self.excluded:
                    return True, v if self.excluded else None
            return False, None
        if dt in ("integer", "double") or self.lo or self.hi:
            if self.lo is None and self.hi is None and not self.excluded:
                return True, None
            eps = 1 if dt == "integer" else 0.01
            lo = (self.lo[0] if self.lo[1] else self.lo[0] + eps) if self.lo else None
            hi = (self.hi[0] if self.hi[1] else self.hi[0] - eps) if self.hi else None
            if dt == "integer":
                lo = None if lo is None else float(int(lo) if lo == int(lo) else int(lo) + 1)
                hi = None if hi is None else float(int(hi) if hi >= 0 else int(hi) - (hi != int(hi)))
            if lo is not None and hi is not None and lo > hi:
                return False, None
            v = lo if lo is not None else (hi if hi is not None else 0.0)
            step = 1.0 if (lo is not None or hi is None) else -1.0
            for _ in range(1000):
                if v not in self.excluded and int(v) not in self.excluded and                         (lo is None or v >= lo) and (hi is None or v <= hi):
                    return True, int(v) if dt == "integer" else v
                v += step
                if (hi is not None and step > 0 and v > hi) or (lo is not None and step < 0 and v < lo):
                    break
            return False, None
        # Free-form string with only exclusions: absent attribute satisfies neq
        return True, None


def solve(conditions: list[Condition]) -> tuple[bool, dict[str, Any]]:
    """Check whether a conjunction of conditions is satisfiable; return a witness assignment."""
    domains: dict[str, _Domain] = {}
    refs: list[Condition] = []
    for c in conditions:
        if isinstance(c.value, AttributeRef):
            refs.append(c)
            continue
        d = domains.setdefault(c.attribute.path, _Domain(c.attribute))
        d.add(c)
    witness: dict[str, Any] = {}
    for path, d in domains.items():
        ok, v = d.solve()
        if not ok:
            return False, {}
        if v is not None:
            witness[path] = v
    for c in refs:
        assert isinstance(c.value, AttributeRef)
        a, b = c.attribute.path, c.value.path
        if c.operator in (Operator.EQ, Operator.IN):
            val = witness.get(a, witness.get(b, "x1"))
            if a in witness and b in witness and witness[a] != witness[b]:
                return False, {}
            witness[a] = witness[b] = val
        else:
            if a in witness and b in witness and witness[a] == witness[b]:
                return False, {}
            witness.setdefault(a, "x1")
            witness.setdefault(b, "x2" if witness[a] != "x2" else "x3")
    return True, witness


def witness_request(witness: dict[str, Any], action: str) -> dict[str, Any]:
    req: dict[str, Any] = {"subject": {}, "action": action, "resource": {}, "environment": {}}
    for path, v in witness.items():
        cat, _, attr = path.partition(".")
        if cat in req and isinstance(req[cat], dict):
            req[cat][attr] = v
    return req


def _action_overlap(a: Rule, b: Rule) -> list[str] | None:
    if not a.actions and not b.actions:
        return ["*"]
    if not a.actions:
        return list(b.actions)
    if not b.actions:
        return list(a.actions)
    common = [x for x in a.actions if x in b.actions]
    return common or None


def implies(premise: list[Condition], conclusion: Condition) -> bool:
    """premise => conclusion  iff  premise AND NOT conclusion is unsatisfiable."""
    ok, _ = solve([*premise, conclusion.negated()])
    return not ok


def subsumes(general: Rule, specific: Rule) -> bool:
    """True if every request matched by `specific` is also matched by `general`."""
    if general.actions and (not specific.actions or not set(specific.actions) <= set(general.actions)):
        return False
    prem = specific.all_conditions()
    if not solve(prem)[0]:
        return True
    return all(implies(prem, c) for c in general.all_conditions())


@dataclass
class Finding:
    kind: str  # conflict | overlap | exception | redundant | unreachable | unsatisfiable | broad
    severity: str  # info | warning | error
    rules: list[str]
    message: str
    witness: dict[str, Any] | None = None
    resolution: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def _incidental(a: Rule, b: Rule) -> bool:
    """True when the overlap only arises because the rules constrain *different* attributes of the
    same entity (e.g. role=nurse vs employment_type=contractor): an intersection rather than a
    direct contradiction."""
    for cat in ("subject", "resource"):
        pa = {c.attribute.path for c in getattr(a, cat)}
        pb = {c.attribute.path for c in getattr(b, cat)}
        if pa and pb and not (pa & pb):
            return True
    return False


def _winner(policy: Policy, a: Rule, b: Rule) -> str:
    alg = policy.combining
    if alg == CombiningAlgorithm.FIRST_APPLICABLE:
        return a.id
    deny_wins = alg in (CombiningAlgorithm.DENY_OVERRIDES, CombiningAlgorithm.PERMIT_UNLESS_DENY)
    want = Effect.DENY if deny_wins else Effect.PERMIT
    return a.id if a.effect == want else b.id


def analyze(policy: Policy) -> list[Finding]:
    """Detect conflicts, intended exceptions, redundant/unreachable and impossible rules."""
    findings: list[Finding] = []
    rules = policy.rules
    for r in rules:
        ok, _ = solve(r.all_conditions())
        if not ok:
            findings.append(Finding("unsatisfiable", "error", [r.id],
                                    f"Rule {r.id} can never apply: its conditions contradict each other."))
        elif not r.all_conditions() and not r.actions:
            findings.append(Finding("broad", "warning", [r.id],
                                    f"Rule {r.id} has no conditions and applies to every request "
                                    f"({r.effect.value} everything)."))
    for i, j in itertools.combinations(range(len(rules)), 2):
        a, b = rules[i], rules[j]
        acts = _action_overlap(a, b)
        if acts is None:
            continue
        ok, wit = solve([*a.all_conditions(), *b.all_conditions()])
        if not ok:
            continue
        action = acts[0] if acts[0] != "*" else "read"
        req = witness_request(wit, action)
        if a.effect != b.effect:
            same_source = a.source_text and a.source_text == b.source_text
            winner = _winner(policy, a, b)
            if same_source:
                findings.append(Finding(
                    "exception", "info", [a.id, b.id],
                    f"{a.id} and {b.id} overlap by design (an exception stated in the same sentence); "
                    f"{winner} wins under {policy.combining.value}.", req, f"{winner} wins"))
            elif _incidental(a, b):
                findings.append(Finding(
                    "overlap", "info", [a.id, b.id],
                    f"{a.id} ({a.effect.value}) and {b.id} ({b.effect.value}) constrain different attributes "
                    f"but both apply to a request like the example; {winner} wins under "
                    f"{policy.combining.value}.", req, f"{winner} wins"))
            else:
                findings.append(Finding(
                    "conflict", "warning", [a.id, b.id],
                    f"{a.id} ({a.effect.value}) and {b.id} ({b.effect.value}) can both apply to the same "
                    f"request; {winner} wins under {policy.combining.value}.", req, f"{winner} wins"))
        else:
            if subsumes(a, b):
                kind = "unreachable" if policy.combining == CombiningAlgorithm.FIRST_APPLICABLE else "redundant"
                findings.append(Finding(kind, "warning", [b.id, a.id],
                                        f"{b.id} is redundant: every request it matches is already matched "
                                        f"by {a.id} with the same effect."))
            elif subsumes(b, a):
                findings.append(Finding("redundant", "warning", [a.id, b.id],
                                        f"{a.id} is redundant: every request it matches is already matched "
                                        f"by {b.id} with the same effect."))
        if policy.combining == CombiningAlgorithm.FIRST_APPLICABLE and a.effect != b.effect and subsumes(a, b):
            findings.append(Finding("unreachable", "warning", [b.id, a.id],
                                    f"{b.id} can never be reached: earlier rule {a.id} matches every "
                                    f"request it would."))
    return findings
