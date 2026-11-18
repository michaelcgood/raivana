"""Policy code generators: IR -> XACML 3.0, NGAC, Cedar, Rego, JSON."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..dictionary import AttributeDictionary
from ..ir import Policy
from .cedar import CedarGenerator
from .ngac import NgacGenerator
from .rego import RegoGenerator
from .xacml import XacmlGenerator, validate_xacml

GENERATORS: dict[str, Any] = {
    "xacml": XacmlGenerator,
    "ngac": NgacGenerator,
    "cedar": CedarGenerator,
    "rego": RegoGenerator,
}
FORMATS = [*GENERATORS, "json"]
EXTENSIONS = {"xacml": ".xml", "ngac": ".pml", "cedar": ".cedar", "rego": ".rego", "json": ".json"}


@dataclass
class Artifact:
    format: str
    content: str
    filename: str
    warnings: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"format": self.format, "content": self.content, "filename": self.filename,
                "warnings": self.warnings, **({"extra": self.extra} if self.extra else {})}


def generate(policy: Policy, fmt: str, dictionary: AttributeDictionary | None = None) -> Artifact:
    fmt = fmt.lower()
    if fmt == "json":
        return Artifact("json", policy.model_dump_json(indent=2), f"{policy.id}.raivana.json")
    if fmt not in GENERATORS:
        raise ValueError(f"Unknown format '{fmt}'. Choose from: {', '.join(FORMATS)}")
    gen = GENERATORS[fmt](dictionary)
    content = gen.generate(policy)
    art = Artifact(fmt, content, f"{policy.id}{EXTENSIONS[fmt]}", list(gen.warnings))
    if fmt == "xacml":
        art.warnings += [f"XACML structure: {e}" for e in validate_xacml(content)]
    if fmt == "ngac" and gen.graph is not None:
        art.extra["graph"] = gen.graph.to_dict()
    return art


def generate_all(policy: Policy, formats: list[str], dictionary: AttributeDictionary | None = None) -> dict[str, Artifact]:
    return {f: generate(policy, f, dictionary) for f in formats}


__all__ = ["Artifact", "FORMATS", "GENERATORS", "generate", "generate_all", "validate_xacml"]
