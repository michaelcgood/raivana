"""Offline (rules engine) translation behaviour - mirrors the patent's FIG. 2/3 pipeline."""

import pytest

from raivana import PolicyComposer, evaluate
from raivana.engines.rules import split_clauses
from raivana.evals import run_eval


@pytest.fixture(scope="module")
def composer():
    return PolicyComposer(engine="rules", learn=False)


def decide(policy, s, a, r, e=None):
    return evaluate(policy, {"subject": s, "action": a, "resource": r, "environment": e or {}}).decision


def test_only_subject_generates_permit_and_deny(composer):
    res = composer.translate("Only doctors can prescribe medication.")
    effects = sorted(r.effect.value for r in res.policy.rules)
    assert effects == ["Deny", "Permit"]
    assert res.sentences[0].template == "only-subject"
    assert decide(res.policy, {"role": "doctor"}, "prescribe", {"type": "prescription"}) == "Permit"
    assert decide(res.policy, {"role": "nurse"}, "prescribe", {"type": "prescription"}) == "Deny"


def test_time_window_and_emergency(composer):
    res = composer.translate("Nurses may view patient records during business hours. "
                             "Nurses can view any patient record in an emergency.")
    p = res.policy
    assert decide(p, {"role": "nurse"}, "read", {"type": "medical_record"}, {"hour": 9}) == "Permit"
    assert decide(p, {"role": "nurse"}, "read", {"type": "medical_record"}, {"hour": 17}) == "NotApplicable"
    assert decide(p, {"role": "nurse"}, "read", {"type": "medical_record"}, {"hour": 3, "emergency": True}) == "Permit"


def test_relations_same_department_and_ownership(composer):
    res = composer.translate("Physicians may update medical records of patients in their department. "
                             "Employees cannot approve their own expense reports.")
    p = res.policy
    assert decide(p, {"role": "doctor", "department": "x"}, "write", {"type": "medical_record", "department": "x"}) == "Permit"
    assert decide(p, {"role": "doctor", "department": "x"}, "write", {"type": "medical_record", "department": "y"}) == "NotApplicable"
    assert decide(p, {"employment_type": "employee", "id": "u1"}, "approve", {"type": "expense_report", "owner": "u1"}) == "Deny"


def test_default_deny_sentence_sets_default(composer):
    res = composer.translate("Managers can approve payments. All other access is denied.")
    assert res.policy.default_effect is not None and res.policy.default_effect.value == "Deny"
    assert decide(res.policy, {"role": "intern"}, "approve", {"type": "transaction"}) == "Deny"


def test_learns_unknown_vocabulary():
    comp = PolicyComposer(engine="rules", learn=True)
    res = comp.translate("Pharmacy technicians can dispense prescriptions.")
    assert any(item["value"] == "pharmacy_technician" for item in res.learned)
    # Claim 1: the candidate attribute is stored in the dictionary and reused.
    assert "pharmacy_technician" in comp.dictionary.attributes["role"].values
    res2 = comp.translate("Pharmacy technicians cannot delete prescriptions.")
    assert not res2.learned


def test_policy_set_vocabulary_is_loaded(composer):
    # Claim 4: 'psychiatrist' is only defined in the healthcare policy set.
    assert "psychiatrist" in composer.dictionary.attributes["role"].values


def test_clause_splitting_carries_subject():
    parts = [p for p, _ in split_clauses("Billing staff can view billing records but cannot view medical records.")]
    assert parts == ["Billing staff can view billing records", "Billing staff cannot view medical records."]


def test_reject_moves_to_next_candidate(composer):
    res = composer.translate("Only doctors can prescribe medication.")
    first = res.sentences[0].template
    res = composer.reject(res, 0)
    assert res.sentences[0].template != first or res.sentences[0].status == "unresolved"


def test_non_rule_sentences_are_skipped(composer):
    res = composer.translate("Hospital Access Policy\nOnly doctors can prescribe medication.")
    assert res.sentences[0].status == "not-a-rule"
    assert res.sentences[1].status == "translated"


def test_offline_eval_suite_core_tier_passes():
    report = run_eval(engine="rules")
    core = report["by_tier"]["core"]
    assert core["passed"] == core["total"], [c for c in report["cases"] if not c["passed"]]
