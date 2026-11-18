"""CLI, REST API, MCP server, archive and Claude engine (with a fake client - no network)."""

import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from raivana import PolicyComposer
from raivana.archive import PolicyArchive
from raivana.cli import main
from raivana.engines.llm import LLMCondition, LLMResult, LLMRule, LLMSentence, ModelRouter


def test_cli_translate_and_test(tmp_path):
    runner = CliRunner()
    out = tmp_path / "out"
    r = runner.invoke(main, ["translate", "-e", "rules", "--policy-id", "t", "-o", str(out), "-F", "all",
                             "Only doctors can prescribe medication."])
    assert r.exit_code == 0, r.output
    assert {p.suffix for p in out.iterdir()} == {".json", ".xml", ".pml", ".cedar", ".rego"}
    r = runner.invoke(main, ["test", str(out / "t.raivana.json"), "-s", "role=nurse", "-a", "prescribe",
                             "-R", "type=prescription", "--json"])
    assert json.loads(r.output)["decision"] == "Deny"


def test_cli_check_and_explain(tmp_path):
    runner = CliRunner()
    runner.invoke(main, ["translate", "-e", "rules", "--policy-id", "c", "-o", str(tmp_path), "-F", "json",
                         "Nurses can read patient records. Contractors cannot read patient records."])
    r = runner.invoke(main, ["check", str(tmp_path / "c.raivana.json"), "--json"])
    assert any(f["kind"] == "overlap" for f in json.loads(r.output)["findings"])
    r = runner.invoke(main, ["explain", str(tmp_path / "c.raivana.json")])
    assert "ALLOW" in r.output and "DENY" in r.output


def test_archive_versions(tmp_path):
    comp = PolicyComposer(engine="rules", learn=False)
    res = comp.translate("Only doctors can prescribe medication.", policy_id="a", formats=["cedar"])
    arch = PolicyArchive(tmp_path)
    m1 = arch.save(res.policy, res.artifacts)
    assert arch.save(res.policy, res.artifacts).get("unchanged")
    res2 = comp.translate("Only nurses can prescribe medication.", policy_id="a")
    m2 = arch.save(res2.policy)
    assert (m1["version"], m2["version"]) == ("v0001", "v0002")
    assert arch.load("a").rules[0].subject[0].values == ["nurse"]


def test_rest_api():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from raivana.server import app

    client = TestClient(app)
    assert "rules" in client.get("/api/health").json()["engines"]
    assert "Raivana" in client.get("/").text or "raivana" in client.get("/").text
    res = client.post("/api/translate", json={"text": "Only doctors can prescribe medication.", "engine": "rules"}).json()
    assert len(res["policy"]["rules"]) == 2 and "cedar" in res["artifacts"]
    d = client.post("/api/evaluate", json={"policy": res["policy"], "request": {
        "subject": {"role": "doctor"}, "action": "prescribe", "resource": {"type": "prescription"}}}).json()
    assert d["decision"] == "Permit"
    after = client.post(f"/api/sessions/{res['session']}/remove", json={"rule_id": "r2"}).json()
    assert len(after["policy"]["rules"]) == 1


def test_mcp_tools():
    pytest.importorskip("mcp")
    from raivana import mcp_server

    out = mcp_server.translate_policy("Only doctors can prescribe medication.", ["cedar"], engine="rules")
    assert out["summary"] and "forbid" in out["code"]["cedar"]
    d = mcp_server.evaluate_access(out["policy"], {"role": "nurse"}, "prescribe", {"type": "prescription"})
    assert d["decision"] == "Deny"


# --------------------------------------------------------------------------- Claude engine, no network
class FakeMessages:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(input_tokens=100, output_tokens=200, cache_read_input_tokens=5000,
                                cache_creation_input_tokens=0)
        return SimpleNamespace(parsed_output=self.result, usage=usage, stop_reason="end_turn", model=kwargs["model"])


class FakeClient:
    def __init__(self, result):
        self.messages = FakeMessages(result)
        self.beta = SimpleNamespace(messages=self.messages)


def _llm_result():
    cond = LLMCondition(category="subject", attribute="role", datatype="string", operator="eq", values=["cfo"],
                        ref_category=None, ref_attribute=None)
    amt = LLMCondition(category="resource", attribute="amount", datatype="integer", operator="gt",
                       values=["50000"], ref_category=None, ref_attribute=None)
    rule = LLMRule(effect="Permit", actions=["approve"], conditions=[cond, amt], explanation="CFO approves big wires")
    return LLMResult(sentences=[LLMSentence(index=0, is_rule=True, confidence=0.9, rules=[rule])],
                     new_attributes=[], default_effect=None, combining=None, ambiguities=["'big' is vague"])


def test_hybrid_sends_only_hard_sentences_to_claude():
    client = FakeClient(_llm_result())
    comp = PolicyComposer(engine="hybrid", llm_client=client, learn=False)
    res = comp.translate("Wire transfers above fifty thousand dollars need sign-off from the CFO. "
                         "Only doctors can prescribe medication.")
    assert len(client.messages.calls) == 1
    user_msg = client.messages.calls[0]["messages"][0]["content"]
    assert "fifty thousand" in user_msg and "prescribe" not in user_msg  # offline handled sentence 2
    assert res.sentences[0].engine == "claude" and res.sentences[1].engine == "rules"
    assert res.policy.rules[0].resource[0].value == 50000
    assert res.usage[0].cache_read_tokens == 5000 and res.cost_usd > 0
    # Prompt caching: the system prompt is marked cacheable.
    assert client.messages.calls[0]["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_model_router():
    r = ModelRouter(mode="auto")
    assert r.choose(["Nurses can read charts."]) == "claude-haiku-5-5"
    long = ["Unless otherwise approved, only managers, except in emergencies, may ..."] * 6
    assert r.choose(long) == "claude-opus-5-5"
    assert ModelRouter().choose(long) == "claude-sonnet-5-5"


def test_env_file_loading(tmp_path, monkeypatch):
    from raivana import config

    (tmp_path / ".env").write_text('# comment\nexport RAIVANA_T1="quoted value"\nRAIVANA_T2=plain # note\n'
                                   "RAIVANA_T3=from-file\nEMPTY=\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAIVANA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("RAIVANA_T3", "from-env")
    for k in ("RAIVANA_T1", "RAIVANA_T2", "EMPTY"):
        monkeypatch.delenv(k, raising=False)
    config.load_env(force=True)
    import os

    assert os.environ["RAIVANA_T1"] == "quoted value"
    assert os.environ["RAIVANA_T2"] == "plain"
    assert os.environ["RAIVANA_T3"] == "from-env"  # real env vars always win
    assert "EMPTY" not in os.environ
    assert config.mask("sk-ant-api03-abcdefghijklmnop") == "sk-ant-...mnop"


def test_policy_id_from_file_name(tmp_path):
    from raivana.composer import slugify

    folder = tmp_path / "hospital"
    folder.mkdir()
    (folder / "policy.txt").write_text("Only doctors can prescribe medication.", encoding="utf-8")
    (tmp_path / "bank-rules.txt").write_text("Tellers cannot approve wire transfers.", encoding="utf-8")
    runner = CliRunner()
    out = tmp_path / "out"
    for f in (folder / "policy.txt", tmp_path / "bank-rules.txt"):
        r = runner.invoke(main, ["translate", "-e", "rules", "-F", "json", "-f", str(f), "-o", str(out)])
        assert r.exit_code == 0, r.output
    assert {p.name for p in out.iterdir()} == {"hospital.raivana.json", "bank-rules.raivana.json"}
    assert slugify("Hospital EHR Access Policy - Only doctors can prescribe medication") == \
        "hospital-ehr-access-policy-only-doctors"
