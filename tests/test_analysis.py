from raivana import AttributeRef, CombiningAlgorithm, Condition, Effect, Operator, Policy, Rule, evaluate
from raivana.analysis import analyze, solve
from raivana.validate import check_rule

ROLE = AttributeRef(category="subject", id="role")
TYPE = AttributeRef(category="resource", id="type")
HOUR = AttributeRef(category="environment", id="hour", datatype="integer")


def c(ref, op, v):
    return Condition(attribute=ref, operator=Operator(op), value=v)


def rule(rid, effect, *conds, actions=("read",), src=""):
    r = Rule(id=rid, effect=Effect(effect), actions=list(actions), source_text=src)
    for cond in conds:
        getattr(r, cond.attribute.category).append(cond)
    return r


def test_solver_detects_contradiction_and_builds_witness():
    assert not solve([c(HOUR, "gte", 17), c(HOUR, "lt", 9)])[0]
    ok, w = solve([c(HOUR, "gte", 9), c(HOUR, "lt", 17), c(ROLE, "neq", "intern")])
    assert ok and 9 <= w["environment.hour"] < 17


def test_combining_algorithms():
    p = Policy(id="p", rules=[rule("a", "Permit", c(ROLE, "eq", "nurse")), rule("b", "Deny", c(TYPE, "eq", "x"))])
    req = {"subject": {"role": "nurse"}, "action": "read", "resource": {"type": "x"}}
    assert evaluate(p, req).decision == "Deny"
    p.combining = CombiningAlgorithm.PERMIT_OVERRIDES
    assert evaluate(p, req).decision == "Permit"
    p.combining = CombiningAlgorithm.FIRST_APPLICABLE
    assert evaluate(p, req).decision == "Permit"
    p.combining = CombiningAlgorithm.DENY_UNLESS_PERMIT
    assert evaluate(p, {"subject": {}, "action": "read"}).decision == "Deny"


def test_missing_attribute_semantics_and_multivalued():
    p = Policy(id="p", rules=[rule("a", "Permit", c(ROLE, "neq", "intern"))])
    assert evaluate(p, {"subject": {}, "action": "read"}).decision == "Permit"
    p2 = Policy(id="p", rules=[rule("a", "Permit", c(ROLE, "eq", "nurse"))])
    assert evaluate(p2, {"subject": {"role": ["clerk", "nurse"]}, "action": "read"}).decision == "Permit"


def test_conflict_and_redundancy_findings():
    p = Policy(id="p", rules=[
        rule("r1", "Permit", c(ROLE, "eq", "nurse")),
        rule("r2", "Deny", c(TYPE, "eq", "audit_log")),
        rule("r3", "Permit", c(ROLE, "eq", "nurse"), c(TYPE, "eq", "chart")),
        rule("r4", "Deny", c(HOUR, "gt", 20), c(HOUR, "lt", 5)),
    ])
    kinds = {(f.kind, tuple(f.rules)) for f in analyze(p)}
    assert ("conflict", ("r1", "r2")) in kinds
    assert ("redundant", ("r3", "r1")) in kinds
    assert ("unsatisfiable", ("r4",)) in kinds
    conflict = next(f for f in analyze(p) if f.kind == "conflict")
    assert conflict.witness["subject"]["role"] == "nurse"


def test_validation_catches_type_errors():
    bad = rule("x", "Permit", c(ROLE, "gt", 3))
    assert any("non-numeric" in i.message for i in check_rule(bad))
