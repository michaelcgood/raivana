"""Differential verification of generated policy code.

Generates randomised access requests from a policy's own attribute domains, then
checks that the decisions of Raivana's reference PDP agree with the decisions of the
real target engines evaluating the generated code:

  * Cedar -- via `cedarpy` (official Cedar engine bindings), if installed
  * Rego  -- via the `opa` CLI, if it is on PATH

This turns "the generated code looks right" into "the generated code provably makes
the same decisions on N requests".
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .analysis import evaluate
from .generators.cedar import CedarGenerator
from .generators.rego import RegoGenerator
from .ir import AttributeRef, CombiningAlgorithm, Effect, Policy


@dataclass
class VerifyReport:
    engine: str
    available: bool
    checked: int = 0
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    skipped_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.available and not self.error and not self.mismatches

    def to_dict(self) -> dict[str, Any]:
        return {"engine": self.engine, "available": self.available, "checked": self.checked,
                "ok": self.ok, "mismatches": self.mismatches[:10], "error": self.error,
                "skipped_reason": self.skipped_reason}


def sample_requests(policy: Policy, n: int = 200, seed: int = 7) -> list[dict[str, Any]]:
    """Random requests drawn from values that appear in the policy plus 'other' values."""
    rng = random.Random(seed)
    domains: dict[str, tuple[AttributeRef, list[Any]]] = {}
    for rule in policy.rules:
        for c in rule.all_conditions():
            refs = [c.attribute] + ([c.value] if isinstance(c.value, AttributeRef) else [])
            for ref in refs:
                _, vals = domains.setdefault(ref.path, (ref, []))
                if ref is c.attribute and not isinstance(c.value, AttributeRef):
                    for v in c.values:
                        if isinstance(v, bool):
                            vals.extend([True, False])
                        elif isinstance(v, (int, float)):
                            vals.extend([v - 1, v, v + 1])
                        else:
                            vals.append(v)
    actions = policy.actions() or ["read"]
    actions = actions + ["other_action"]
    shared_pool = ["u1", "u2", "eng", "fin"]
    out = []
    for _ in range(n):
        req: dict[str, Any] = {"subject": {}, "action": rng.choice(actions), "resource": {}, "environment": {}}
        for ref, vals in domains.values():
            if rng.random() < 0.15:
                continue  # attribute absent
            if vals:
                pool = vals + (["zz_other"] if ref.datatype == "string" else [])
                v = rng.choice(pool)
            elif ref.datatype == "integer":
                v = rng.randint(0, 23)
            elif ref.datatype == "boolean":
                v = rng.random() < 0.5
            else:
                v = rng.choice(shared_pool)
            req[ref.category][ref.id] = v
        out.append(req)
    return out


# --------------------------------------------------------------------------- Cedar
def _cedar_value(v: Any) -> Any:
    return v


def cedar_decide(cedar_text: str, request: dict[str, Any]) -> str:
    import cedarpy  # type: ignore[import-not-found]

    entities = [
        {"uid": {"type": "User", "id": "subject"},
         "attrs": {k: _cedar_value(v) for k, v in request["subject"].items()}, "parents": []},
        {"uid": {"type": "Resource", "id": "resource"},
         "attrs": {k: _cedar_value(v) for k, v in request["resource"].items()}, "parents": []},
        {"uid": {"type": "Action", "id": request["action"]}, "attrs": {}, "parents": []},
    ]
    req = {"principal": 'User::"subject"', "action": f'Action::"{request["action"]}"',
           "resource": 'Resource::"resource"', "context": request["environment"]}
    res = cedarpy.is_authorized(req, cedar_text, entities)
    errors = getattr(getattr(res, "diagnostics", None), "errors", None)
    if errors:
        raise RuntimeError(f"Cedar evaluation errors: {errors}")
    return "Permit" if res.allowed else "Deny"


def verify_cedar(policy: Policy, requests: list[dict[str, Any]]) -> VerifyReport:
    rep = VerifyReport("cedar", available=True)
    try:
        import cedarpy  # noqa: F401
    except ImportError:
        rep.available, rep.skipped_reason = False, "cedarpy not installed (pip install cedarpy)"
        return rep
    if policy.combining not in (CombiningAlgorithm.DENY_OVERRIDES, CombiningAlgorithm.DENY_UNLESS_PERMIT) \
            or policy.default_effect == Effect.PERMIT:
        rep.skipped_reason = "Cedar only expresses deny-overrides with default deny"
        return rep
    text = CedarGenerator().generate(policy)
    for req in requests:
        expected = evaluate(policy, req).decision
        expected = "Permit" if expected == "Permit" else "Deny"
        try:
            got = cedar_decide(text, req)
        except Exception as e:  # noqa: BLE001 - surface any engine error
            rep.error = str(e)
            return rep
        rep.checked += 1
        if got != expected:
            rep.mismatches.append({"request": req, "raivana": expected, "cedar": got})
    return rep


# --------------------------------------------------------------------------- Rego / OPA
def verify_rego(policy: Policy, requests: list[dict[str, Any]], opa: str | None = None) -> VerifyReport:
    rep = VerifyReport("rego", available=True)
    opa = opa or shutil.which("opa")
    if not opa:
        rep.available, rep.skipped_reason = False, "opa CLI not on PATH (https://www.openpolicyagent.org)"
        return rep
    gen = RegoGenerator(package="raivana.verify")
    text = gen.generate(policy)
    with tempfile.TemporaryDirectory() as td:
        pol = Path(td) / "policy.rego"
        pol.write_text(text, encoding="utf-8")
        inputs = Path(td) / "inputs.json"
        inputs.write_text(json.dumps({"requests": requests}), encoding="utf-8")
        # Evaluate every request in one OPA call by mapping the module over the inputs.
        query = ("[d | some r in input.requests; "
                 "d := data.raivana.verify.decision with input as r]")
        proc = subprocess.run([opa, "eval", "-f", "json", "-d", str(pol), "-i", str(inputs), query],
                              capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            rep.error = proc.stderr.strip() or proc.stdout.strip()
            return rep
        decisions = json.loads(proc.stdout)["result"][0]["expressions"][0]["value"]
    for req, got in zip(requests, decisions, strict=True):
        expected = evaluate(policy, req).decision
        rep.checked += 1
        if got != expected:
            rep.mismatches.append({"request": req, "raivana": expected, "rego": got})
    return rep


def verify(policy: Policy, n: int = 200, seed: int = 7) -> list[VerifyReport]:
    reqs = sample_requests(policy, n, seed)
    return [verify_cedar(policy, reqs), verify_rego(policy, reqs)]
