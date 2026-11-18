"""Attribute dictionary (US 10,606,957 B1, element 106).

Links words and phrases found in natural-language policy text to the attributes
(and attribute values) used in the generated access control code.

The dictionary can be populated three ways, mirroring the patent claims:
  * from bundled / organisation-specific dictionary packs (YAML),
  * from the natural language policy sets (claims 4, 11, 18) -- see `learn_from_policy_sets`,
  * from user input (claims 5, 12, 19) -- see `add_value` / `add_attribute`,
and, during translation, candidate attributes discovered by NLP are stored back into
it (claim 1) -- see `RulesEngine` and `learn`.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from .ir import AttributeRef, Category, DataType
from .nlp import stem, tokenize

DEFAULT_XACML_IDS = {
    ("subject", "id"): "urn:oasis:names:tc:xacml:1.0:subject:subject-id",
    ("subject", "role"): "urn:oasis:names:tc:xacml:2.0:subject:role",
    ("resource", "id"): "urn:oasis:names:tc:xacml:1.0:resource:resource-id",
    ("action", "action"): "urn:oasis:names:tc:xacml:1.0:action:action-id",
}


@dataclass
class AttributeDef:
    id: str
    categories: list[Category]
    datatype: DataType = "string"
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    values: dict[str, list[str]] = field(default_factory=dict)
    xacml_id: str | None = None
    domain: str = "core"
    learned: bool = False

    def ref(self, category: Category | None = None) -> AttributeRef:
        return AttributeRef(category=category or self.categories[0], id=self.id, datatype=self.datatype)

    def xacml(self, category: Category) -> str:
        if self.xacml_id:
            return self.xacml_id
        return DEFAULT_XACML_IDS.get((category, self.id), f"urn:raivana:attribute:{category}:{self.id}")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "categories": list(self.categories), "datatype": self.datatype}
        if self.description:
            d["description"] = self.description
        if self.aliases:
            d["aliases"] = self.aliases
        if self.values:
            d["values"] = self.values
        if self.xacml_id:
            d["xacml_id"] = self.xacml_id
        if self.learned:
            d["learned"] = True
        return d


@dataclass
class Mention:
    """A dictionary hit: what a phrase in the text means."""

    kind: str  # "value" | "action" | "attribute" | "universal" | "nobody"
    attribute: AttributeDef | None = None
    value: str | None = None
    surface: str = ""
    explicit: bool = True


class AttributeDictionary:
    def __init__(self) -> None:
        self.attributes: dict[str, AttributeDef] = {}
        self.actions: dict[str, list[str]] = {}
        self.action_domains: dict[str, str] = {}
        self.universal_subjects: list[str] = []
        self.nobody_subjects: list[str] = []
        self.domain_keywords: dict[str, set[str]] = {}
        self._index: dict[tuple[str, ...], list[Mention]] | None = None
        self._max_len = 1
        self.learned_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ loading
    @classmethod
    def default(cls, extra_paths: Iterable[str | os.PathLike[str]] = ()) -> AttributeDictionary:
        d = cls()
        pkg = resources.files("raivana") / "data" / "dictionary"
        for entry in sorted(pkg.iterdir(), key=lambda p: p.name):
            if entry.name.endswith(".yaml"):
                d.load_yaml_text(entry.read_text(encoding="utf-8"))
        for p in extra_paths:
            if Path(p).exists():
                d.load_file(p)
        return d

    def load_file(self, path: str | os.PathLike[str]) -> None:
        self.load_yaml_text(Path(path).read_text(encoding="utf-8"))

    def load_yaml_text(self, text: str) -> None:
        data = yaml.safe_load(text) or {}
        domain = data.get("domain", "core")
        kws = self.domain_keywords.setdefault(domain, set())
        for kw in data.get("keywords", []):
            kws.add(stem(kw))
        for raw in data.get("attributes", []):
            cats = raw.get("categories") or [raw.get("category", "subject")]
            attr = AttributeDef(
                id=raw["id"],
                categories=list(cats),
                datatype=raw.get("datatype", "string"),
                description=raw.get("description", ""),
                aliases=list(raw.get("aliases", [])),
                values={str(k): list(v or []) for k, v in (raw.get("values") or {}).items()},
                xacml_id=raw.get("xacml_id"),
                domain=domain,
                learned=bool(raw.get("learned", False)),
            )
            self._merge_attribute(attr)
            for surfaces in attr.values.values():
                for s in surfaces:
                    kws.add(stem(s.split()[-1]))
        for action, surfaces in (data.get("actions") or {}).items():
            self.action_domains.setdefault(str(action), domain)
            existing = self.actions.setdefault(str(action), [])
            for s in surfaces or []:
                if s not in existing:
                    existing.append(s)
        self.universal_subjects += [s for s in data.get("universal_subjects", []) if s not in self.universal_subjects]
        self.nobody_subjects += [s for s in data.get("nobody_subjects", []) if s not in self.nobody_subjects]
        self._index = None

    def _merge_attribute(self, attr: AttributeDef) -> None:
        cur = self.attributes.get(attr.id)
        if cur is None:
            self.attributes[attr.id] = attr
            return
        for c in attr.categories:
            if c not in cur.categories:
                cur.categories.append(c)
        for a in attr.aliases:
            if a not in cur.aliases:
                cur.aliases.append(a)
        for val, surfaces in attr.values.items():
            bucket = cur.values.setdefault(val, [])
            for s in surfaces:
                if s not in bucket:
                    bucket.append(s)
        cur.xacml_id = cur.xacml_id or attr.xacml_id
        cur.description = cur.description or attr.description

    # ------------------------------------------------------------------ lookup
    @staticmethod
    def key(phrase: str) -> tuple[str, ...]:
        return tuple(t.stem for t in tokenize(phrase) if t.is_word or t.lower[:1].isdigit())

    def _build_index(self) -> None:
        idx: dict[tuple[str, ...], list[Mention]] = {}

        def add(phrase: str, m: Mention) -> None:
            k = self.key(phrase)
            if k:
                idx.setdefault(k, []).append(m)

        for attr in self.attributes.values():
            for value, surfaces in attr.values.items():
                add(value.replace("_", " "), Mention("value", attr, value, value, explicit=False))
                for s in surfaces:
                    add(s, Mention("value", attr, value, s))
            for alias in [attr.id.replace("_", " "), *attr.aliases]:
                add(alias, Mention("attribute", attr, None, alias))
        for action, surfaces in self.actions.items():
            if action != "*":
                add(action, Mention("action", None, action, action))
            for s in surfaces:
                add(s, Mention("action", None, action, s))
        for s in self.universal_subjects:
            add(s, Mention("universal", surface=s))
        for s in self.nobody_subjects:
            add(s, Mention("nobody", surface=s))
        # Values beat attribute names beat actions for the same phrase.
        order = {"value": 0, "nobody": 1, "universal": 2, "action": 3, "attribute": 4}
        for k in idx:
            idx[k].sort(key=lambda m: (order[m.kind], not m.explicit))
        self._index = idx
        self._max_len = max((len(k) for k in idx), default=1)

    def lookup(self, stems: tuple[str, ...]) -> list[Mention]:
        if self._index is None:
            self._build_index()
        assert self._index is not None
        return self._index.get(stems, [])

    @property
    def max_phrase_len(self) -> int:
        if self._index is None:
            self._build_index()
        return self._max_len

    def known_verb_stems(self) -> set[str]:
        out = set()
        for action, surfaces in self.actions.items():
            for s in [action, *surfaces]:
                k = self.key(s)
                if k:
                    out.add(k[0])
        out.discard("*")
        return out

    def known_noun_stems(self) -> set[str]:
        out = set()
        for attr in self.attributes.values():
            for surfaces in attr.values.values():
                for s in surfaces:
                    k = self.key(s)
                    if k:
                        out.add(k[-1])
        return out

    def get(self, attr_id: str) -> AttributeDef | None:
        return self.attributes.get(attr_id)

    def action_names(self) -> list[str]:
        return [a for a in self.actions if a != "*"]

    def xacml_id(self, ref: AttributeRef) -> str:
        attr = self.attributes.get(ref.id)
        if attr:
            return attr.xacml(ref.category)
        return DEFAULT_XACML_IDS.get((ref.category, ref.id), f"urn:raivana:attribute:{ref.category}:{ref.id}")

    def detect_policy_type(self, text: str) -> tuple[str, dict[str, int]]:
        """Identify the policy type (FIG. 2, step 204) by domain vocabulary overlap."""
        stems = {t.stem for t in tokenize(text) if t.is_word}
        scores = {dom: len(stems & kws) for dom, kws in self.domain_keywords.items() if dom != "core"}
        if not scores or max(scores.values()) == 0:
            return "general", scores
        return max(scores, key=lambda k: scores[k]), scores

    # ------------------------------------------------------------------ learning
    def add_attribute(self, attr_id: str, category: Category, datatype: DataType = "string",
                      description: str = "", source: str = "user") -> AttributeDef:
        attr = self.attributes.get(attr_id)
        if attr is None:
            attr = AttributeDef(id=attr_id, categories=[category], datatype=datatype,
                                description=description, domain="learned", learned=True)
            self.attributes[attr_id] = attr
            self.learned_log.append({"type": "attribute", "id": attr_id, "category": category, "source": source})
        elif category not in attr.categories:
            attr.categories.append(category)
        self._index = None
        return attr

    def add_value(self, attr_id: str, value: str, surfaces: Iterable[str] = (),
                  category: Category = "subject", source: str = "user") -> None:
        attr = self.attributes.get(attr_id) or self.add_attribute(attr_id, category, source=source)
        bucket = attr.values.setdefault(value, [])
        new = [s for s in surfaces if s not in bucket]
        bucket.extend(new)
        self.learned_log.append({"type": "value", "attribute": attr_id, "value": value,
                                 "surfaces": new, "source": source})
        self._index = None

    def add_action(self, action: str, surfaces: Iterable[str] = (), source: str = "user") -> None:
        self.action_domains.setdefault(action, "learned")
        bucket = self.actions.setdefault(action, [])
        bucket.extend(s for s in surfaces if s not in bucket)
        self.learned_log.append({"type": "action", "action": action, "source": source})
        self._index = None

    def learn_from_policy_sets(self, policy_sets: Any) -> int:
        """Populate the dictionary from attributes declared in policy-set samples (claim 4)."""
        n = 0
        for sample in policy_sets.samples:
            for attr in sample.attributes:
                self.add_value(attr["id"], attr["value"], attr.get("surfaces", []),
                               category=attr.get("category", "subject"), source=f"policy-set:{sample.id}")
                n += 1
        return n

    # ------------------------------------------------------------------ export
    def learned_yaml(self) -> str:
        attrs = [a.to_dict() for a in self.attributes.values() if a.learned]
        partial: dict[str, dict[str, Any]] = {}
        for entry in self.learned_log:
            if entry["type"] == "value":
                a = self.attributes[entry["attribute"]]
                if a.learned:
                    continue
                p = partial.setdefault(a.id, {"id": a.id, "categories": list(a.categories),
                                              "datatype": a.datatype, "values": {}})
                p["values"].setdefault(entry["value"], []).extend(entry["surfaces"])
        actions = {e["action"]: self.actions[e["action"]] for e in self.learned_log if e["type"] == "action"}
        doc: dict[str, Any] = {"domain": "learned", "attributes": attrs + list(partial.values())}
        if actions:
            doc["actions"] = actions
        return yaml.safe_dump(doc, sort_keys=False)

    def save_learned(self, path: str | os.PathLike[str]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.learned_yaml(), encoding="utf-8")

    def to_prompt(self) -> str:
        """Compact, deterministic listing for the LLM engine (stable across calls => cacheable)."""
        lines = ["ATTRIBUTES (category.id : datatype : canonical values = synonyms)"]
        for attr in sorted(self.attributes.values(), key=lambda a: a.id):
            cats = "|".join(attr.categories)
            vals = "; ".join(
                f"{v}=" + ",".join(s[:40] for s in surf[:4]) for v, surf in sorted(attr.values.items())
            )
            lines.append(f"- {cats}.{attr.id} : {attr.datatype}" + (f" : {vals}" if vals else ""))
        lines.append("ACTIONS (canonical = synonyms; '*' means any action)")
        for action, surf in sorted(self.actions.items()):
            lines.append(f"- {action} = " + ",".join(surf[:6]))
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        return {
            "attributes": [a.to_dict() | {"domain": a.domain} for a in self.attributes.values()],
            "actions": self.actions,
            "domains": sorted(self.domain_keywords),
        }
