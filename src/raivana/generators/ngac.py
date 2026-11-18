"""NGAC generator (US 10,606,957 B1, claims 3, 10, 17; FIG. 6).

Next Generation Access Control (ANSI INCITS 565 / NIST) expresses policy as a graph:
users and objects are assigned to *attributes*, attributes are contained in
*policy classes*, *associations* grant access rights from a user attribute to an
object attribute, and *prohibitions* deny them.

Translation (FIG. 6, step 608):
  * equality / membership subject conditions  -> user attributes   ("role=doctor")
  * equality / membership resource conditions -> object attributes ("type=medical_record")
  * several conditions on one side            -> an intersection attribute assigned to each
  * Permit                                     -> association(UA, ops, OA)
  * Deny                                       -> prohibition(UA, ops, OA)
Statements lacking enough parameters to be complete and valid in NGAC (environment
conditions, numeric thresholds, negated subjects) are split out as `unsupported`
(FIG. 6, step 610) so they can be enforced as obligations or by the PEP.

Output: a PML (Policy Machine Language) style script and a JSON graph.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..explain import describe_rule
from ..ir import AttributeRef, Condition, Effect, Operator, Policy, Rule

ALL_USERS = "all_users"
ALL_OBJECTS = "all_objects"


@dataclass
class NgacGraph:
    policy_class: str
    user_attributes: dict[str, list[str]] = field(default_factory=dict)   # node -> parents
    object_attributes: dict[str, list[str]] = field(default_factory=dict)
    associations: list[dict[str, Any]] = field(default_factory=list)
    prohibitions: list[dict[str, Any]] = field(default_factory=list)
    unsupported: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_class": self.policy_class,
            "user_attributes": [{"name": k, "assigned_to": v} for k, v in self.user_attributes.items()],
            "object_attributes": [{"name": k, "assigned_to": v} for k, v in self.object_attributes.items()],
            "associations": self.associations,
            "prohibitions": self.prohibitions,
            "unsupported": self.unsupported,
        }


def _node(c: Condition) -> list[str]:
    return [f"{c.attribute.id}={v}" for v in c.values]


class NgacGenerator:
    format = "ngac"
    extension = ".pml"
    title = "NGAC (PML)"

    def __init__(self, dictionary: Any = None) -> None:
        self.warnings: list[str] = []
        self.graph: NgacGraph | None = None

    def _side(self, graph: NgacGraph, conds: list[Condition], kind: str, rule: Rule,
              negations: list[Condition]) -> list[str]:
        """Return the attribute node(s) (alternatives, OR-ed) representing a set of conditions."""
        store = graph.user_attributes if kind == "user" else graph.object_attributes
        top = ALL_USERS if kind == "user" else ALL_OBJECTS
        store.setdefault(top, [graph.policy_class])
        positive: list[list[str]] = []
        for c in conds:
            if isinstance(c.value, AttributeRef):
                graph.unsupported.append({"rule": rule.id, "condition": c.describe(),
                                          "reason": "attribute-to-attribute relation (enforce via PEP/obligation)"})
                continue
            if c.operator in (Operator.EQ, Operator.IN) and c.attribute.datatype != "boolean":
                nodes = _node(c)
                for n in nodes:
                    store.setdefault(n, [top])
                positive.append(nodes)
            elif c.operator in (Operator.EQ, Operator.IN):
                n = f"{c.attribute.id}={str(c.values[0]).lower()}"
                store.setdefault(n, [top])
                positive.append([n])
            elif c.operator in (Operator.NEQ, Operator.NOT_IN):
                negations.append(c)
            else:
                graph.unsupported.append({"rule": rule.id, "condition": c.describe(),
                                          "reason": "numeric threshold (enforce via PEP/obligation)"})
        if not positive:
            return [top]
        # cartesian product of alternatives -> intersection nodes
        combos: list[list[str]] = [[]]
        for alts in positive:
            combos = [c + [a] for c in combos for a in alts]
        out = []
        for combo in combos:
            if len(combo) == 1:
                out.append(combo[0])
            else:
                name = " & ".join(sorted(combo))
                store.setdefault(name, sorted(combo))
                out.append(name)
        return out

    def build(self, policy: Policy) -> NgacGraph:
        g = NgacGraph(policy_class=policy.id)
        for r in policy.rules:
            neg_s: list[Condition] = []
            neg_r: list[Condition] = []
            uas = self._side(g, r.subject, "user", r, neg_s)
            oas = self._side(g, r.resource, "object", r, neg_r)
            for c in r.environment:
                g.unsupported.append({"rule": r.id, "condition": c.describe(),
                                      "reason": "environment condition (enforce via PEP/obligation)"})
            ops = list(r.actions) or ["*"]
            if r.effect == Effect.PERMIT:
                if neg_s or neg_r:
                    for c in neg_s + neg_r:
                        g.unsupported.append({"rule": r.id, "condition": c.describe(),
                                              "reason": "negated condition in a permission"})
                for ua in uas:
                    for oa in oas:
                        g.associations.append({"rule": r.id, "user_attribute": ua, "access_rights": ops,
                                               "object_attribute": oa})
            else:
                if neg_s:
                    for c in neg_s:
                        g.unsupported.append({
                            "rule": r.id, "condition": c.describe(),
                            "reason": "negated subject in a prohibition; covered by NGAC default deny "
                                      "only when no association grants the access"})
                    continue
                complement = [f"!{n}" for c in neg_r for n in _node(c)]
                for n in complement:
                    g.object_attributes.setdefault(n[1:], [ALL_OBJECTS])
                for ua in uas:
                    g.prohibitions.append({
                        "name": f"{r.id}-{ua}".replace(" ", ""), "rule": r.id, "user_attribute": ua,
                        "access_rights": ops,
                        "intersection": [*([] if oas == [ALL_OBJECTS] and complement else oas), *complement]})
        self.graph = g
        return g

    def generate(self, policy: Policy) -> str:
        self.warnings = []
        g = self.build(policy)
        q = json.dumps
        out = [f"// NGAC policy (PML) generated by Raivana for policy '{policy.id}'",
               f"// Fingerprint {policy.fingerprint()}. NGAC is deny-by-default: access requires an association.",
               "", f"create policy class {q(g.policy_class)}", ""]
        for name, parents in g.user_attributes.items():
            out.append(f"create user attribute {q(name)} in [{', '.join(q(p) for p in parents)}]")
        out.append("")
        for name, parents in g.object_attributes.items():
            out.append(f"create object attribute {q(name)} in [{', '.join(q(p) for p in parents)}]")
        out.append("")
        rules = {r.id: r for r in policy.rules}
        for a in g.associations:
            out.append(f"// {a['rule']}: {describe_rule(rules[a['rule']])}")
            out.append(f"associate {q(a['user_attribute'])} and {q(a['object_attribute'])} "
                       f"with [{', '.join(q(x) for x in a['access_rights'])}]")
        if g.prohibitions:
            out.append("")
        for p in g.prohibitions:
            targets = ", ".join(("!" + q(t[1:])) if t.startswith("!") else q(t) for t in p["intersection"])
            out.append(f"// {p['rule']}: {describe_rule(rules[p['rule']])}")
            out.append(f"create prohibition {q(p['name'])} deny user attribute {q(p['user_attribute'])} "
                       f"access rights [{', '.join(q(x) for x in p['access_rights'])}] "
                       f"on intersection of [{targets}]")
        if g.unsupported:
            out.append("")
            out.append("// Not expressible as NGAC assignments/associations; enforce as obligations or at the PEP:")
            for u in g.unsupported:
                out.append(f"//   [{u['rule']}] {u['condition']} -- {u['reason']}")
            self.warnings.append(f"{len(g.unsupported)} condition(s) could not be expressed natively in NGAC.")
        return "\n".join(out) + "\n"
