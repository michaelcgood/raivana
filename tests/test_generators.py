import json
import xml.etree.ElementTree as ET

import pytest

from raivana import PolicyComposer, generate
from raivana.generators import validate_xacml
from raivana.verify import sample_requests, verify_cedar

HOSPITAL = """Only doctors can prescribe medication.
Nurses may view patient records during business hours.
Physicians may update medical records of patients in their department.
Contractors cannot access patient records unless they are on the corporate network.
Employees cannot approve their own expense reports.
DevOps engineers can deploy to production only between 10pm and 6am.
Users with clearance level 3 or higher can read restricted documents.
Nobody may delete audit logs.
All other access is denied."""


@pytest.fixture(scope="module")
def policy():
    return PolicyComposer(engine="rules", learn=False).translate(HOSPITAL, policy_id="hospital").policy


def test_xacml_is_structurally_valid(policy):
    xml = generate(policy, "xacml").content
    assert validate_xacml(xml) == []
    root = ET.fromstring(xml.encode())
    ns = "{urn:oasis:names:tc:xacml:3.0:core:schema:wd-17}"
    assert root.tag == f"{ns}PolicySet"
    assert root.get("PolicyCombiningAlgId").endswith("deny-unless-permit")  # default Deny
    assert len(root.findall(f".//{ns}Rule")) == len(policy.rules)


def test_xacml_validator_rejects_broken_documents():
    assert validate_xacml("<not-xml")
    bad = '<Policy xmlns="urn:oasis:names:tc:xacml:3.0:core:schema:wd-17" PolicyId="p" Version="1"/>'
    errors = validate_xacml(bad)
    assert any("RuleCombiningAlgId" in e for e in errors) and any("Target" in e for e in errors)


def test_cedar_output_shape(policy):
    cedar = generate(policy, "cedar").content
    assert cedar.count("permit (") + cedar.count("forbid (") == len(policy.rules)
    assert 'principal has role' in cedar and "resource.owner == principal.id" in cedar


def test_rego_output_shape(policy):
    rego = generate(policy, "rego").content
    assert "package raivana.hospital" in rego
    assert 'default decision := "Deny"' in rego
    assert rego.count(" if {") >= len(policy.rules)


def test_ngac_graph(policy):
    art = generate(policy, "ngac")
    g = art.extra["graph"]
    assert any(a["user_attribute"] == "role=doctor" for a in g["associations"])
    assert g["unsupported"], "environment conditions must be reported, not silently dropped"
    assert "create policy class" in art.content


def test_json_roundtrip(policy):
    from raivana import Policy

    assert Policy.model_validate(json.loads(generate(policy, "json").content)) == policy


def test_cedar_engine_agrees_with_reference_pdp(policy):
    pytest.importorskip("cedarpy")
    rep = verify_cedar(policy, sample_requests(policy, 400))
    assert rep.error is None, rep.error
    assert rep.checked == 400 and not rep.mismatches, rep.mismatches[:2]


def test_rego_engine_agrees_with_reference_pdp(policy):
    import shutil

    if not shutil.which("opa"):
        pytest.skip("opa CLI not installed")
    from raivana.verify import verify_rego

    rep = verify_rego(policy, sample_requests(policy, 400))
    assert rep.error is None, rep.error
    assert not rep.mismatches, rep.mismatches[:2]
