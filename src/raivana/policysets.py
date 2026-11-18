"""Natural language policy sets (US 10,606,957 B1, element 108).

A policy set is a collection of sample rules, organised by language and by
policy type / organisation / department. During translation, candidate rules
found in the input text are matched against these samples (FIG. 3, step 304),
the best sample is populated with the candidate attributes (step 306), and the
result is checked for well-formedness (step 308).
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from .ir import Condition, Effect, Rule

# Markers whose presence/absence changes the meaning of a rule. A mismatch on any
# of these is penalised much more heavily than a missing optional slot.
KEY_MARKERS = {"PERMIT", "DENY", "REQUIRE", "NOBODY", "ONLY_SUBJ", "ONLY_COND", "ONLY_ACTION",
               "UNLESS", "EXCEPT"}
GROUPS = ("S", "R", "C", "U", "X")


@dataclass
class Sample:
    id: str
    example: str
    signature: list[str]
    produces: list[dict[str, Any]]
    language: str = "en"
    policy_type: str = "general"
    organization: str | None = None
    attributes: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Candidate:
    sample: Sample
    score: float  # structural similarity of signatures, 0..1

    @property
    def id(self) -> str:
        return self.sample.id


class PolicySets:
    def __init__(self, samples: Iterable[Sample] = ()) -> None:
        self.samples: list[Sample] = list(samples)

    @classmethod
    def default(cls, extra_paths: Iterable[str | os.PathLike[str]] = ()) -> PolicySets:
        ps = cls()
        pkg = resources.files("raivana") / "data" / "policysets"
        for entry in sorted(pkg.iterdir(), key=lambda p: p.name):
            if entry.name.endswith(".yaml"):
                ps.load_yaml_text(entry.read_text(encoding="utf-8"))
        for p in extra_paths:
            path = Path(p)
            files = sorted(path.glob("*.yaml")) if path.is_dir() else [path]
            for f in files:
                ps.load_yaml_text(f.read_text(encoding="utf-8"))
        return ps

    def load_yaml_text(self, text: str) -> None:
        data = yaml.safe_load(text) or {}
        for raw in data.get("samples", []):
            self.samples.append(Sample(
                id=raw["id"],
                example=raw.get("example", ""),
                signature=list(raw["signature"]),
                produces=list(raw["produces"]),
                language=data.get("language", "en"),
                policy_type=data.get("policy_type", "general"),
                organization=data.get("organization"),
                attributes=list(raw.get("attributes", [])),
            ))

    def select(self, language: str = "en", policy_type: str | None = None,
               organization: str | None = None) -> list[Sample]:
        """Select the policy sets relevant to the input (FIG. 2, step 206)."""
        out = []
        for s in self.samples:
            if s.language != language:
                continue
            if s.organization and organization and s.organization != organization:
                continue
            if s.policy_type in ("general", policy_type) or s.organization == organization:
                out.append(s)
        return out

    def match(self, signature: list[str], samples: list[Sample]) -> list[Candidate]:
        """Rank sample rules by similarity to a sentence signature (FIG. 3, step 304)."""
        cands = []
        sig_keys = set(signature) & KEY_MARKERS
        seen_sigs: set[tuple[str, ...]] = set()
        for s in samples:
            ratio = SequenceMatcher(None, signature, s.signature, autojunk=False).ratio()
            mismatch = len(sig_keys ^ (set(s.signature) & KEY_MARKERS))
            score = ratio * (0.45 ** mismatch)
            cands.append(Candidate(s, round(score, 4)))
        cands.sort(key=lambda c: (-c.score, c.sample.policy_type != "general"))
        # De-duplicate samples that are structurally identical (keep the best ranked).
        unique = []
        for c in cands:
            key = (tuple(c.sample.signature), repr(c.sample.produces))
            if key in seen_sigs:
                continue
            seen_sigs.add(key)
            unique.append(c)
        return unique


def _merge(conds: list[Condition]) -> list[Condition]:
    """AND-merge conditions. Same-attribute equality on different values inside one
    group was already OR-ed upstream; here a later equality on the same attribute
    overrides an earlier one (exception semantics, e.g. 'employees ... except contractors')."""
    out: list[Condition] = []
    for c in conds:
        replaced = False
        for i, existing in enumerate(out):
            if existing.attribute.path == c.attribute.path and existing.operator.value in ("eq", "in") \
                    and c.operator.value in ("eq", "in") and existing.values != c.values:
                out[i] = c
                replaced = True
                break
            if existing.model_dump() == c.model_dump():
                replaced = True
                break
        if not replaced:
            out.append(c)
    return out


def populate(sample: Sample, groups: dict[str, list[Condition]], actions: list[str],
             other_actions: list[str]) -> list[Rule]:
    """Populate a sample rule with candidate attributes (FIG. 3, step 306).

    Returns IR rules without ids; NOT_ groups are expanded with De Morgan's law."""
    rules: list[Rule] = []
    for spec in sample.produces:
        effect = Effect(spec["effect"])
        positive: list[Condition] = []
        alternatives: list[list[Condition]] = []
        skip = False
        for g in spec.get("when", []):
            if g.startswith("NOT_"):
                conds = groups.get(g[4:], [])
                if not conds:
                    skip = True  # cannot negate an empty group
                    break
                alternatives.append([c.negated() for c in conds])
            else:
                positive.extend(groups.get(g, []))
        if skip:
            continue
        acts = other_actions if spec.get("actions") == "OTHER" else actions
        if spec.get("actions") == "OTHER" and not acts:
            continue
        for combo in itertools.product(*alternatives) if alternatives else [()]:
            conds = _merge([*positive, *combo])
            rule = Rule(id="", effect=effect, actions=[a for a in acts if a != "*"], template=sample.id)
            for c in conds:
                cat = c.attribute.category
                if cat == "subject":
                    rule.subject.append(c)
                elif cat == "resource":
                    rule.resource.append(c)
                elif cat == "environment":
                    rule.environment.append(c)
            rules.append(rule)
    return rules
