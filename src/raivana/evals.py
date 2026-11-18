"""Behavioural evaluation harness: policy text -> generated policy -> expected decisions."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from .analysis import evaluate
from .composer import PolicyComposer


def load_cases(path: str | None = None) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8") if path else \
        (resources.files("raivana") / "data" / "evals" / "cases.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)["cases"]


def _request(check: dict[str, Any]) -> dict[str, Any]:
    return {"subject": check.get("s", {}), "action": check["a"], "resource": check.get("r", {}),
            "environment": check.get("e", {})}


def run_eval(engine: str = "rules", model: str | None = None, cases_path: str | None = None,
             progress: bool = False, composer: PolicyComposer | None = None) -> dict[str, Any]:
    cases = load_cases(cases_path)
    out: list[dict[str, Any]] = []
    cost = 0.0
    checks = checks_passed = 0
    for case in cases:
        # Fresh composer per case so learned vocabulary cannot leak between cases.
        comp = composer or PolicyComposer(engine=engine, model=model, learn=False)
        result = comp.translate(case["text"], policy_id=case["id"])
        cost += result.cost_usd
        failures = []
        for chk in case["checks"]:
            d = evaluate(result.policy, _request(chk))
            allowed = d.decision == "Permit"
            checks += 1
            if allowed == bool(chk["allow"]):
                checks_passed += 1
            else:
                failures.append(f"{_request(chk)} -> {d.decision}, expected "
                                f"{'Permit' if chk['allow'] else 'not Permit'}")
        if not result.policy.rules:
            failures.insert(0, "no rules were produced (untranslated)")
        out.append({"id": case["id"], "tier": case.get("tier", "core"), "text": case["text"],
                    "passed": not failures, "failures": failures, "rules": len(result.policy.rules),
                    "engines": sorted({s.engine for s in result.sentences if s.engine})})
        if progress:
            print("." if not failures else "F", end="", flush=True)
    if progress:
        print()
    passed = sum(1 for c in out if c["passed"])
    by_tier: dict[str, dict[str, int]] = {}
    for c in out:
        t = by_tier.setdefault(c["tier"], {"passed": 0, "total": 0})
        t["total"] += 1
        t["passed"] += c["passed"]
    return {"engine": engine, "model": model, "total": len(out), "passed": passed,
            "checks": checks, "checks_passed": checks_passed,
            "accuracy": checks_passed / checks if checks else 0.0,
            "by_tier": by_tier, "cost_usd": round(cost, 6), "cases": out}
