"""Policy composer (US 10,606,957 B1, element 104).

Orchestrates the full translation pipeline:

  FIG. 2  rule identification: language (202) -> policy type (204) -> policy sets (206)
          -> candidate attributes (208) -> candidate rules (210)
  FIG. 3  rule translation: match samples (304) -> populate (306) -> well-formed? (308)
          -> generate code (310) -> user confirmation (312, reject => back to 304) -> export (314)

Engines:
  rules   offline NLP + sample-rule matching; free, deterministic
  claude  every sentence translated by Claude (structured IR output)
  hybrid  offline first; only low-confidence sentences are sent to Claude (cheapest LLM mode)
  auto    hybrid when an API key is configured, otherwise rules
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import nlp
from .analysis import Finding, analyze
from .dictionary import AttributeDictionary
from .engines.llm import ClaudeEngine, LLMUsage, ModelRouter, llm_available, resolve_model
from .engines.rules import RulesEngine, SentenceAnalysis, detect_combining, split_clauses
from .explain import describe_rule
from .generators import Artifact, generate_all
from .ir import CombiningAlgorithm, Effect, Policy, Rule
from .policysets import PolicySets
from .validate import Issue, check_policy, is_well_formed

ENGINES = ("auto", "rules", "hybrid", "claude")


@dataclass
class SentenceResult:
    index: int
    text: str
    offset: int
    status: str  # translated | unresolved | not-a-rule | meta
    engine: str | None = None
    confidence: float = 0.0
    template: str | None = None
    candidate_index: int = 0
    rule_ids: list[str] = field(default_factory=list)
    analysis: SentenceAnalysis | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self, detail: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "index": self.index, "text": self.text, "offset": self.offset, "status": self.status,
            "engine": self.engine, "confidence": self.confidence, "template": self.template,
            "candidate_index": self.candidate_index, "rule_ids": self.rule_ids, "notes": self.notes,
        }
        if detail and self.analysis is not None:
            d["analysis"] = self.analysis.to_dict()
        return d


@dataclass
class TranslationResult:
    policy: Policy
    sentences: list[SentenceResult]
    language: str
    language_confidence: float
    policy_type: str
    issues: list[Issue] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    learned: list[dict[str, Any]] = field(default_factory=list)
    usage: list[LLMUsage] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    engine: str = "rules"

    @property
    def cost_usd(self) -> float:
        return round(sum(u.cost_usd for u in self.usage), 6)

    @property
    def unresolved(self) -> list[SentenceResult]:
        return [s for s in self.sentences if s.status == "unresolved"]

    def summary(self) -> list[dict[str, Any]]:
        return [{"id": r.id, "effect": r.effect.value, "summary": describe_rule(r),
                 "confidence": r.confidence, "engine": r.engine, "source": r.source_text}
                for r in self.policy.rules]

    def to_dict(self, detail: bool = True) -> dict[str, Any]:
        return {
            "policy": self.policy.model_dump(mode="json"),
            "summary": self.summary(),
            "sentences": [s.to_dict(detail) for s in self.sentences],
            "language": self.language, "language_confidence": self.language_confidence,
            "policy_type": self.policy_type, "engine": self.engine,
            "issues": [i.to_dict() for i in self.issues],
            "findings": [f.to_dict() for f in self.findings],
            "artifacts": {k: a.to_dict() for k, a in self.artifacts.items()},
            "learned": self.learned,
            "usage": [u.to_dict() for u in self.usage], "cost_usd": self.cost_usd,
            "ambiguities": self.ambiguities,
        }


_RULEISH = re.compile(r"\b(?:need|needs|require|requires|required|allowed|permitted|approval|approve|"
                      r"sign-off|access|get to|gets to|only|nobody|no one|never|prohibited|forbidden|"
                      r"restricted|permission|authori[sz]ed|entitled|off-limits|off limits|except|"
                      r"unless|hands off|keep|limited to|reserved for|banned|blocked)\b", re.I)


def _maybe_rule(sentence: str) -> bool:
    """Cheap pre-filter: is a modal-less sentence worth sending to Claude?"""
    words = sentence.split()
    return len(words) >= 4 and bool(_RULEISH.search(sentence))


def slugify(text: str, default: str = "policy") -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:40] or default


class PolicyComposer:
    def __init__(self, dictionary: AttributeDictionary | None = None, policy_sets: PolicySets | None = None,
                 engine: str = "auto", model: str | None = None, route: str = "fixed",
                 threshold: float = 0.6, hybrid_threshold: float = 0.85, learn: bool = True, effort: str = "medium",
                 provider: str | None = None, extra_dictionaries: Iterable[str] = (),
                 extra_policy_sets: Iterable[str] = (), llm_client: Any = None) -> None:
        if engine not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}")
        user_dict = Path(os.environ.get("RAIVANA_HOME", Path.home() / ".raivana")) / "dictionary.yaml"
        self.dictionary = dictionary or AttributeDictionary.default(
            [*extra_dictionaries, *([user_dict] if user_dict.exists() else [])])
        self.policy_sets = policy_sets or PolicySets.default(extra_policy_sets)
        # Claim 4: populate the attribute dictionary from the natural language policy sets.
        self.dictionary.learn_from_policy_sets(self.policy_sets)
        self.dictionary.learned_log = [e for e in self.dictionary.learned_log
                                       if not str(e.get("source", "")).startswith("policy-set:")]
        self.engine_name = engine
        self.threshold = threshold
        self.hybrid_threshold = hybrid_threshold
        self.learn = learn
        self.rules_engine = RulesEngine(self.dictionary, self.policy_sets, learn=learn)
        self.router = ModelRouter(model=resolve_model(model), mode="auto" if route == "auto" else "fixed")
        self.effort = effort
        self.provider = provider
        self._llm_client = llm_client
        self._claude: ClaudeEngine | None = None

    # ------------------------------------------------------------------ engines
    @property
    def effective_engine(self) -> str:
        if self.engine_name == "auto":
            return "hybrid" if (self._llm_client is not None or llm_available()) else "rules"
        return self.engine_name

    def claude(self) -> ClaudeEngine:
        if self._claude is None:
            self._claude = ClaudeEngine(self.dictionary, self.policy_sets, self.router, self.effort,
                                        self.provider, self._examples(), client=self._llm_client)
        return self._claude

    def _examples(self) -> list[tuple[str, str]]:
        """Few-shot examples generated from the policy sets themselves (deterministic => cacheable)."""
        out = []
        fixed = RulesEngine(self.dictionary, self.policy_sets, learn=False)  # policy_type fixed => stable prompt
        for sample in self.policy_sets.samples:
            if sample.language != "en":
                continue
            an = fixed.analyze(sample.example)
            if not an.is_rule:
                continue
            cand = next((c for c in self.policy_sets.match(an.signature, [sample])), None)
            if cand is None:
                continue
            rules = fixed.rules_for(an, cand)
            ir = "; ".join(f"{r.effect.value} actions={r.actions or ['*']} if "
                           + (" AND ".join(c.describe() for c in r.all_conditions()) or "true")
                           for r in rules)
            out.append((sample.example, ir))
        return out

    # ------------------------------------------------------------------ translate
    def translate(self, text: str, policy_id: str | None = None, formats: Iterable[str] = ("json",),
                  organization: str | None = None, description: str = "") -> TranslationResult:
        language, lconf = nlp.detect_language(text)
        policy_type, _ = self.dictionary.detect_policy_type(text)
        self.rules_engine.policy_type = policy_type
        samples = self.policy_sets.select(language, policy_type, organization)
        engine = self.effective_engine
        if language != "en" and engine == "rules":
            # Offline engine is English-only; still try, but flag it.
            pass

        hint, _evidence = detect_combining(text)
        policy = Policy(id=policy_id or slugify(description or text.split(".")[0]),
                        description=description, language=language, policy_type=policy_type,
                        source_text=text)
        if hint in ("deny-unless-permit",):
            policy.default_effect = Effect.DENY
        elif hint == "permit-unless-deny":
            policy.default_effect = Effect.PERMIT
        elif hint:
            policy.combining = CombiningAlgorithm(hint)

        result = TranslationResult(policy=policy, sentences=[], language=language, language_confidence=lconf,
                                   policy_type=policy_type, engine=engine)
        per_sentence: dict[int, list[Rule]] = {}
        to_llm: list[tuple[int, str]] = []

        units = [(clause, off + coff) for sent, off in nlp.split_sentences(text)
                 for clause, coff in split_clauses(sent)]
        for idx, (sent, off) in enumerate(units):
            sr = SentenceResult(index=idx, text=sent, offset=off, status="unresolved")
            result.sentences.append(sr)
            if engine == "claude" or language != "en" and engine in ("hybrid",):
                to_llm.append((idx, sent))
                continue
            an = self.rules_engine.analyze(sent, off)
            sr.analysis = an
            sr.notes = list(an.notes)
            if an.meta:
                sr.status, sr.engine = "meta", "rules"
                continue
            if not an.is_rule:
                sr.status, sr.engine = "not-a-rule", "rules"
                if engine == "hybrid" and _maybe_rule(sent):
                    to_llm.append((idx, sent))  # let Claude decide (e.g. "X needs sign-off from Y")
                continue
            self.rules_engine.candidates(an, samples)
            hybrid = engine == "hybrid"
            if hybrid and an.learned:
                # Offline engine had to guess vocabulary: Claude does this better.
                to_llm.append((idx, sent))
                continue
            rules = self._accept_candidate(an, sr, 0, self.hybrid_threshold if hybrid else self.threshold)
            if rules is not None:
                per_sentence[idx] = rules
                self._commit_learned(an.learned, result)
            elif hybrid:
                to_llm.append((idx, sent))

        if to_llm:
            llm = self.claude().translate(to_llm, context=f"policy type: {policy_type}")
            result.usage.extend(llm.usage)
            result.ambiguities.extend(llm.ambiguities)
            for err in llm.errors:
                result.issues.append(Issue("warning", None, f"Claude engine: {err}"))
            for idx, sent in to_llm:
                sr = result.sentences[idx]
                if idx in llm.non_rules:
                    sr.status, sr.engine = "not-a-rule", "claude"
                    continue
                rules = llm.rules.get(idx)
                if not rules:
                    sr.notes.append("Claude engine returned no rules for this sentence.")
                    continue
                conf = llm.confidence.get(idx, 0.8)
                for k, r in enumerate(rules):
                    r.source_text, r.confidence = sent, conf
                    r.id = r.id or f"tmp{k}"
                if all(is_well_formed(r) for r in rules):
                    per_sentence[idx] = rules
                    sr.status, sr.engine, sr.confidence = "translated", "claude", conf
                    sr.template = None
                else:
                    sr.notes.append("Claude output failed the well-formedness check.")
            if llm.default_effect and policy.default_effect is None:
                policy.default_effect = Effect(llm.default_effect)
            if llm.combining and policy.combining == CombiningAlgorithm.DENY_OVERRIDES:
                policy.combining = CombiningAlgorithm(llm.combining)
            self._learn_from_llm(llm.new_attributes, result)

        self._assemble(result, per_sentence)
        self.finalize(result, formats)
        return result

    def _accept_candidate(self, an: SentenceAnalysis, sr: SentenceResult, start: int,
                          threshold: float | None = None) -> list[Rule] | None:
        """Try candidates from `start` on; return rules of the first well-formed one above threshold."""
        threshold = self.threshold if threshold is None else threshold
        for ci in range(start, len(an.candidates)):
            cand = an.candidates[ci]
            conf = an.confidence(cand)
            if conf < threshold:
                break
            rules = self.rules_engine.rules_for(an, cand)
            for k, r in enumerate(rules):
                r.id = r.id or f"tmp{k}"
            if rules and all(is_well_formed(r) for r in rules):
                sr.status, sr.engine, sr.confidence = "translated", "rules", conf
                sr.template, sr.candidate_index = cand.id, ci
                return rules
        sr.status = "unresolved"
        if an.candidates:
            best = an.candidates[0]
            sr.notes.append(f"Best sample '{best.id}' scored {an.confidence(best)} "
                            f"(threshold {threshold}).")
        return None

    def _assemble(self, result: TranslationResult, per_sentence: dict[int, list[Rule]]) -> None:
        rules: list[Rule] = []
        n = 0
        for sr in result.sentences:
            sr.rule_ids = []
            for r in per_sentence.get(sr.index, []):
                n += 1
                r.id = f"r{n}"
                rules.append(r)
                sr.rule_ids.append(r.id)
        result.policy.rules = rules

    def _commit_learned(self, learned: list[dict[str, Any]], result: TranslationResult) -> None:
        """Store candidate attributes discovered by NLP in the attribute dictionary (claim 1)."""
        for item in learned:
            result.learned.append({k: v for k, v in item.items() if k not in ("attr_id", "category")})
            if not self.learn:
                continue
            if item["kind"] == "action":
                self.dictionary.add_action(item["value"], [item["surface"].lower()], source="nlp")
            else:
                self.dictionary.add_value(item["attr_id"], item["value"], [item["surface"]],
                                          category=item["category"], source="nlp")

    def _learn_from_llm(self, attrs: list[dict[str, Any]], result: TranslationResult) -> None:
        for a in attrs:
            entry = {"kind": "action" if a["category"] == "action" else "value",
                     "attribute": None if a["category"] == "action" else f"{a['category']}.{a['attribute']}",
                     "value": a.get("value") or a["attribute"], "surface": ", ".join(a.get("synonyms", [])),
                     "source": "claude"}
            result.learned.append(entry)
            if not self.learn:
                continue
            if a["category"] == "action":
                self.dictionary.add_action(a.get("value") or a["attribute"], a.get("synonyms", []), source="claude")
            elif a.get("value"):
                self.dictionary.add_value(a["attribute"], a["value"], a.get("synonyms", []),
                                          category=a["category"], source="claude")
            else:
                self.dictionary.add_attribute(a["attribute"], a["category"], a.get("datatype", "string"),
                                              source="claude")

    def finalize(self, result: TranslationResult, formats: Iterable[str] = ("json",)) -> TranslationResult:
        """(Re)validate, analyse and generate code for the current policy."""
        result.issues = [i for i in result.issues if i.message.startswith("Claude engine")]
        result.issues += check_policy(result.policy, self.dictionary)
        for sr in result.unresolved:
            result.issues.append(Issue("warning", None, f"Sentence {sr.index} was not translated: \"{sr.text}\""))
        result.findings = analyze(result.policy)
        result.artifacts = generate_all(result.policy, list(dict.fromkeys(formats)), self.dictionary)
        return result

    # ------------------------------------------------------------------ review loop (FIG. 3, 312)
    def reject(self, result: TranslationResult, sentence_index: int,
               formats: Iterable[str] = ("json",)) -> TranslationResult:
        """User rejected a sentence's translation: try the next-best sample rule (back to step 304)."""
        sr = result.sentences[sentence_index]
        per_sentence = self._current(result)
        if sr.analysis is None or not sr.analysis.candidates:
            per_sentence.pop(sentence_index, None)
            sr.status = "unresolved"
        else:
            rules = self._accept_candidate(sr.analysis, sr, sr.candidate_index + 1)
            if rules is None:
                per_sentence.pop(sentence_index, None)
                sr.notes.append("No further candidate sample rules; sentence left unresolved.")
            else:
                per_sentence[sentence_index] = rules
        self._assemble(result, per_sentence)
        return self.finalize(result, formats)

    def remove_rule(self, result: TranslationResult, rule_id: str,
                    formats: Iterable[str] = ("json",)) -> TranslationResult:
        """User removed a component of the translation (FIG. 7, step 718)."""
        per_sentence = self._current(result)
        for idx, rules in per_sentence.items():
            per_sentence[idx] = [r for r in rules if r.id != rule_id]
        self._assemble(result, per_sentence)
        return self.finalize(result, formats)

    @staticmethod
    def _current(result: TranslationResult) -> dict[int, list[Rule]]:
        by_id = {r.id: r for r in result.policy.rules}
        return {sr.index: [by_id[i] for i in sr.rule_ids if i in by_id] for sr in result.sentences if sr.rule_ids}

    def save_learned(self, path: str | os.PathLike[str] | None = None) -> Path:
        p = Path(path) if path else Path(os.environ.get("RAIVANA_HOME", Path.home() / ".raivana")) / "dictionary.yaml"
        if p.exists():
            existing = AttributeDictionary()
            existing.load_file(p)
            for a in existing.attributes.values():
                for v, s in a.values.items():
                    self.dictionary.add_value(a.id, v, s, category=a.categories[0], source="file")
        self.dictionary.save_learned(p)
        return p
