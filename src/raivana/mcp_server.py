"""Model Context Protocol server: lets AI agents author, test and analyse access policy.

    raivana mcp                      # stdio transport (Claude Desktop, Claude Code, IDEs)
    raivana mcp --transport streamable-http

Example Claude Code registration:
    claude mcp add raivana -- raivana mcp
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from . import __version__
from .analysis import analyze, evaluate
from .composer import PolicyComposer
from .explain import describe_rule
from .generators import FORMATS, generate
from .ir import Policy
from .validate import check_policy

server = MCPServer(
    "raivana",
    version=__version__,
    instructions=(
        "Raivana translates natural-language access control policy into XACML 3.0, NGAC, Cedar and Rego, "
        "and can test access requests against a policy. Typical flow: translate_policy -> inspect the "
        "plain-English summary and findings -> evaluate_access with concrete requests -> generate_policy_code. "
        "Policies are passed between tools as the `policy` JSON object returned by translate_policy."
    ),
)


def _policy(policy: dict[str, Any] | str) -> Policy:
    return Policy.model_validate_json(policy) if isinstance(policy, str) else Policy.model_validate(policy)


@server.tool()
def translate_policy(text: str, formats: list[str] | None = None, engine: str = "auto",
                     policy_id: str | None = None) -> dict[str, Any]:
    """Translate natural-language access policy into a logical access control policy.

    Args:
        text: Policy statements in plain language, e.g. "Only doctors can prescribe medication."
        formats: Output formats to generate: any of xacml, ngac, cedar, rego, json. Default: cedar.
        engine: "rules" (offline), "claude", "hybrid", or "auto".
        policy_id: Optional identifier for the policy.

    Returns the policy IR (`policy`), a plain-English `summary` of every rule, generated `code`,
    conflict/redundancy `findings`, validation `issues`, and untranslated sentences.
    """
    fmts = [f for f in (formats or ["cedar"]) if f in FORMATS]
    composer = PolicyComposer(engine=engine)
    res = composer.translate(text, policy_id=policy_id, formats=fmts)
    return {
        "policy": res.policy.model_dump(mode="json"),
        "summary": [f"{r.id}: {describe_rule(r)}" for r in res.policy.rules],
        "code": {k: a.content for k, a in res.artifacts.items()},
        "findings": [f.message for f in res.findings],
        "issues": [i.message for i in res.issues],
        "untranslated": [s.text for s in res.unresolved],
        "cost_usd": res.cost_usd,
    }


@server.tool()
def evaluate_access(policy: dict[str, Any], subject: dict[str, Any], action: str,
                    resource: dict[str, Any] | None = None,
                    environment: dict[str, Any] | None = None) -> dict[str, Any]:
    """Decide whether a request is permitted by a policy (returned by translate_policy).

    Args:
        policy: The `policy` object from translate_policy.
        subject: Subject attributes, e.g. {"role": "nurse", "department": "cardiology"}.
        action: Action name, e.g. "read".
        resource: Resource attributes, e.g. {"type": "medical_record"}.
        environment: Environment attributes, e.g. {"hour": 14, "network": "corporate"}.
    """
    d = evaluate(_policy(policy), {"subject": subject, "action": action, "resource": resource or {},
                                   "environment": environment or {}})
    return {"decision": d.decision, "reason": d.reason, "determining_rules": d.determining_rules,
            "applicable_rules": [t.rule_id for t in d.trace if t.applicable]}


@server.tool()
def analyze_policy(policy: dict[str, Any]) -> dict[str, Any]:
    """Validate a policy and detect conflicting, redundant or impossible rules (with example requests)."""
    pol = _policy(policy)
    comp = PolicyComposer(engine="rules")
    return {"issues": [i.to_dict() for i in check_policy(pol, comp.dictionary)],
            "findings": [f.to_dict() for f in analyze(pol)]}


@server.tool()
def generate_policy_code(policy: dict[str, Any], format: str = "cedar") -> str:
    """Generate policy code (xacml, ngac, cedar, rego or json) from a policy IR object."""
    comp = PolicyComposer(engine="rules")
    return generate(_policy(policy), format, comp.dictionary).content


@server.tool()
def list_vocabulary() -> str:
    """List the attribute dictionary: known attributes, canonical values, synonyms and actions."""
    return PolicyComposer(engine="rules").dictionary.to_prompt()


@server.resource("raivana://examples/{name}")
def example(name: str) -> str:
    """Bundled example policies: hospital, bank, cloud."""
    from importlib import resources

    if name not in ("hospital", "bank", "cloud"):
        raise ValueError("unknown example")
    return (resources.files("raivana") / "data" / "examples" / f"{name}.txt").read_text(encoding="utf-8")


def main() -> None:  # pragma: no cover
    server.run("stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
