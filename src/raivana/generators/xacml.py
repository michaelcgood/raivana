"""XACML 3.0 generator (US 10,606,957 B1, claims 2, 9, 16; FIG. 5).

Rules are organised into a PolicySet -> Policy -> Rule hierarchy (FIG. 5, step 510)
and emitted as an XML document importable into XACML 3.0 compliant PDPs (step 512).

Semantics deliberately mirror Raivana's built-in PDP:
  * equality / membership -> <Target> Match (no match when the attribute is absent)
  * negation              -> not(is-in(...)) (true when the attribute is absent)
  * ordering              -> any-of(...)      (false when the attribute is absent)
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from ..dictionary import AttributeDictionary
from ..explain import describe_rule
from ..ir import AttributeRef, CombiningAlgorithm, Condition, Effect, Operator, Policy, Rule

NS = "urn:oasis:names:tc:xacml:3.0:core:schema:wd-17"
F1 = "urn:oasis:names:tc:xacml:1.0:function:"
F3 = "urn:oasis:names:tc:xacml:3.0:function:"
XSD = "http://www.w3.org/2001/XMLSchema#"

CATEGORY = {
    "subject": "urn:oasis:names:tc:xacml:1.0:subject-category:access-subject",
    "resource": "urn:oasis:names:tc:xacml:3.0:attribute-category:resource",
    "action": "urn:oasis:names:tc:xacml:3.0:attribute-category:action",
    "environment": "urn:oasis:names:tc:xacml:3.0:attribute-category:environment",
}
ACTION_ID = "urn:oasis:names:tc:xacml:1.0:action:action-id"

RULE_ALG = {
    CombiningAlgorithm.DENY_OVERRIDES: "urn:oasis:names:tc:xacml:3.0:rule-combining-algorithm:deny-overrides",
    CombiningAlgorithm.PERMIT_OVERRIDES: "urn:oasis:names:tc:xacml:3.0:rule-combining-algorithm:permit-overrides",
    CombiningAlgorithm.FIRST_APPLICABLE: "urn:oasis:names:tc:xacml:1.0:rule-combining-algorithm:first-applicable",
    CombiningAlgorithm.DENY_UNLESS_PERMIT: "urn:oasis:names:tc:xacml:3.0:rule-combining-algorithm:deny-unless-permit",
    CombiningAlgorithm.PERMIT_UNLESS_DENY: "urn:oasis:names:tc:xacml:3.0:rule-combining-algorithm:permit-unless-deny",
}
POLICY_ALG = {k: v.replace("rule-combining", "policy-combining") for k, v in RULE_ALG.items()}

# attribute OP value  ==  any-of(F, value, bag)  where F(value, x) holds
ANY_OF_FN = {Operator.GT: "less-than", Operator.GTE: "less-than-or-equal",
             Operator.LT: "greater-than", Operator.LTE: "greater-than-or-equal"}


def _q(tag: str) -> str:
    return f"{{{NS}}}{tag}"


def _literal(v: Any, datatype: str) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


class XacmlGenerator:
    format = "xacml"
    extension = ".xml"
    title = "XACML 3.0"

    def __init__(self, dictionary: AttributeDictionary | None = None) -> None:
        self.dictionary = dictionary
        self.warnings: list[str] = []

    # -------------------------------------------------------------- helpers
    def _attr_id(self, ref: AttributeRef) -> str:
        if self.dictionary is not None:
            return self.dictionary.xacml_id(ref)
        return f"urn:raivana:attribute:{ref.category}:{ref.id}"

    def _designator(self, parent: ET.Element, ref: AttributeRef) -> ET.Element:
        return ET.SubElement(parent, _q("AttributeDesignator"), {
            "AttributeId": self._attr_id(ref), "Category": CATEGORY[ref.category],
            "DataType": XSD + ref.datatype, "MustBePresent": "false"})

    @staticmethod
    def _value(parent: ET.Element, v: Any, datatype: str) -> ET.Element:
        el = ET.SubElement(parent, _q("AttributeValue"), {"DataType": XSD + datatype})
        el.text = _literal(v, datatype)
        return el

    def _apply(self, parent: ET.Element, fn: str) -> ET.Element:
        return ET.SubElement(parent, _q("Apply"), {"FunctionId": fn})

    def _match_anyof(self, target: ET.Element, ref: AttributeRef, values: list[Any]) -> None:
        any_of = ET.SubElement(target, _q("AnyOf"))
        for v in values:
            all_of = ET.SubElement(any_of, _q("AllOf"))
            m = ET.SubElement(all_of, _q("Match"), {"MatchId": f"{F1}{ref.datatype}-equal"})
            self._value(m, v, ref.datatype)
            self._designator(m, ref)

    def _action_anyof(self, target: ET.Element, actions: list[str]) -> None:
        any_of = ET.SubElement(target, _q("AnyOf"))
        for a in actions:
            all_of = ET.SubElement(any_of, _q("AllOf"))
            m = ET.SubElement(all_of, _q("Match"), {"MatchId": f"{F1}string-equal"})
            self._value(m, a, "string")
            ET.SubElement(m, _q("AttributeDesignator"), {
                "AttributeId": ACTION_ID, "Category": CATEGORY["action"],
                "DataType": XSD + "string", "MustBePresent": "false"})

    def _expression(self, parent: ET.Element, c: Condition) -> None:
        dt = c.attribute.datatype
        op = c.operator
        if isinstance(c.value, AttributeRef):
            if op in (Operator.EQ, Operator.IN, Operator.NEQ, Operator.NOT_IN):
                host = self._apply(parent, f"{F1}not") if op in (Operator.NEQ, Operator.NOT_IN) else parent
                ap = self._apply(host, f"{F1}{dt}-at-least-one-member-of")
                self._designator(ap, c.attribute)
                self._designator(ap, c.value)
            else:
                # attribute OP other-attribute: any-of-any(F, bagA, bagB)
                fn = {Operator.GT: "greater-than", Operator.GTE: "greater-than-or-equal",
                      Operator.LT: "less-than", Operator.LTE: "less-than-or-equal"}[op]
                ap = self._apply(parent, f"{F3}any-of-any")
                ET.SubElement(ap, _q("Function"), {"FunctionId": f"{F1}{dt}-{fn}"})
                self._designator(ap, c.attribute)
                self._designator(ap, c.value)
            return
        if op in (Operator.EQ, Operator.IN, Operator.NEQ, Operator.NOT_IN):
            host = self._apply(parent, f"{F1}not") if op in (Operator.NEQ, Operator.NOT_IN) else parent
            if len(c.values) == 1:
                ap = self._apply(host, f"{F1}{dt}-is-in")
                self._value(ap, c.values[0], dt)
                self._designator(ap, c.attribute)
            else:
                ap = self._apply(host, f"{F1}{dt}-at-least-one-member-of")
                bag = self._apply(ap, f"{F1}{dt}-bag")
                for v in c.values:
                    self._value(bag, v, dt)
                self._designator(ap, c.attribute)
            return
        ap = self._apply(parent, f"{F3}any-of")
        ET.SubElement(ap, _q("Function"), {"FunctionId": f"{F1}{dt}-{ANY_OF_FN[op]}"})
        self._value(ap, c.values[0], dt)
        self._designator(ap, c.attribute)

    def _rule(self, parent: ET.Element, rule: Rule) -> None:
        el = ET.SubElement(parent, _q("Rule"), {"RuleId": rule.id, "Effect": rule.effect.value})
        desc = ET.SubElement(el, _q("Description"))
        desc.text = describe_rule(rule) + (f" Source: \"{rule.source_text}\"" if rule.source_text else "")
        target = ET.SubElement(el, _q("Target"))
        cond_list: list[Condition] = []
        for c in rule.all_conditions():
            if not isinstance(c.value, AttributeRef) and c.operator in (Operator.EQ, Operator.IN):
                self._match_anyof(target, c.attribute, c.values)
            else:
                cond_list.append(c)
        if rule.actions:
            self._action_anyof(target, rule.actions)
        if cond_list:
            cond = ET.SubElement(el, _q("Condition"))
            host = self._apply(cond, f"{F1}and") if len(cond_list) > 1 else cond
            for c in cond_list:
                self._expression(host, c)

    # -------------------------------------------------------------- public
    def generate(self, policy: Policy) -> str:
        self.warnings = []
        ET.register_namespace("", NS)
        if policy.default_effect == Effect.DENY:
            ps_alg = POLICY_ALG[CombiningAlgorithm.DENY_UNLESS_PERMIT]
        elif policy.default_effect == Effect.PERMIT:
            ps_alg = POLICY_ALG[CombiningAlgorithm.PERMIT_UNLESS_DENY]
        else:
            ps_alg = POLICY_ALG[CombiningAlgorithm.DENY_OVERRIDES]
        root = ET.Element(_q("PolicySet"), {
            "PolicySetId": f"urn:raivana:policyset:{policy.id}", "Version": "1.0",
            "PolicyCombiningAlgId": ps_alg})
        d = ET.SubElement(root, _q("Description"))
        d.text = f"Generated by Raivana from natural language policy. Fingerprint {policy.fingerprint()}."
        ET.SubElement(root, _q("Target"))
        pol = ET.SubElement(root, _q("Policy"), {
            "PolicyId": f"urn:raivana:policy:{policy.id}", "Version": "1.0",
            "RuleCombiningAlgId": RULE_ALG[policy.combining]})
        pd = ET.SubElement(pol, _q("Description"))
        pd.text = policy.description or f"Policy {policy.id}"
        ET.SubElement(pol, _q("Target"))
        for r in policy.rules:
            self._rule(pol, r)
        ET.indent(root, space="  ")
        return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


# =========================================================================== structural validation
_KNOWN_FN_PREFIXES = (F1, F3)
_CHILD_ORDER = {
    "PolicySet": ["Description", "PolicyIssuer", "PolicySetDefaults", "Target", "*"],
    "Policy": ["Description", "PolicyIssuer", "PolicyDefaults", "Target", "*"],
    "Rule": ["Description", "Target", "Condition", "ObligationExpressions", "AdviceExpressions"],
}
_REQUIRED_ATTRS = {
    "PolicySet": ["PolicySetId", "Version", "PolicyCombiningAlgId"],
    "Policy": ["PolicyId", "Version", "RuleCombiningAlgId"],
    "Rule": ["RuleId", "Effect"],
    "Match": ["MatchId"],
    "AttributeDesignator": ["AttributeId", "Category", "DataType", "MustBePresent"],
    "AttributeValue": ["DataType"],
    "Apply": ["FunctionId"],
}


def validate_xacml(xml_text: str, xsd_path: str | None = None) -> list[str]:
    """Structural XACML 3.0 checks; full XSD validation when `xsd_path` is given and lxml is installed."""
    errors: list[str] = []
    try:
        root = ET.fromstring(xml_text.encode("utf-8"))
    except ET.ParseError as e:
        return [f"XML is not well-formed: {e}"]

    def local(tag: str) -> str:
        return tag.split("}", 1)[-1]

    for el in root.iter():
        name = local(el.tag)
        if not el.tag.startswith(f"{{{NS}}}"):
            errors.append(f"<{name}> is not in the XACML 3.0 namespace")
        for a in _REQUIRED_ATTRS.get(name, []):
            if a not in el.attrib:
                errors.append(f"<{name}> is missing required attribute {a}")
        if name in _CHILD_ORDER:
            order = _CHILD_ORDER[name]
            last = -1
            for child in el:
                cn = local(child.tag)
                idx = order.index(cn) if cn in order else (order.index("*") if "*" in order else -2)
                if idx == -2:
                    errors.append(f"<{cn}> is not allowed inside <{name}>")
                elif idx < last:
                    errors.append(f"<{cn}> is out of order inside <{name}>")
                last = max(last, idx)
            if name in ("Policy", "PolicySet") and not any(local(c.tag) == "Target" for c in el):
                errors.append(f"<{name}> requires a <Target>")
        if name == "Rule" and el.get("Effect") not in ("Permit", "Deny"):
            errors.append(f"Rule {el.get('RuleId')} has invalid Effect {el.get('Effect')}")
        if name == "Condition" and len(list(el)) != 1:
            errors.append("<Condition> must contain exactly one expression")
        if name == "Match":
            kids = [local(c.tag) for c in el]
            if kids[:1] != ["AttributeValue"] or len(kids) != 2:
                errors.append("<Match> must contain an AttributeValue followed by a designator")
        if name in ("Apply", "Function", "Match"):
            fn = el.get("FunctionId") or el.get("MatchId") or ""
            if not fn.startswith(_KNOWN_FN_PREFIXES):
                errors.append(f"Unknown function {fn}")
    if xsd_path:
        try:
            from lxml import etree  # type: ignore[import-not-found]
        except ImportError:  # pragma: no cover
            errors.append("lxml is not installed; XSD validation skipped")
        else:
            schema = etree.XMLSchema(etree.parse(xsd_path))
            doc = etree.fromstring(xml_text.encode("utf-8"))
            if not schema.validate(doc):
                errors.extend(str(e) for e in schema.error_log)
    return errors
