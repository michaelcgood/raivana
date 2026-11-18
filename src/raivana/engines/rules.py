"""Deterministic (offline) translation engine.

Pipeline per sentence -- a literal implementation of US 10,606,957 B1:
  1. pre-process: extract time windows, numeric limits and attribute relations
  2. NLP: tokenize, lemmatize/stem, POS-tag, morphologically segment          (FIG. 5, 502)
  3. named entity recognition against the attribute dictionary                 (claims 6, 13, 20)
  4. learn unknown candidate attributes and store them in the dictionary      (claim 1)
  5. derive the sentence's semantic signature and match sample rules          (FIG. 3, 304)
  6. populate the best sample rule with the candidate attributes              (FIG. 3, 306)
Well-formedness checking (308) happens in `raivana.validate`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .. import nlp
from ..dictionary import AttributeDef, AttributeDictionary
from ..ir import AttributeRef, Condition, Operator, Rule
from ..policysets import Candidate, PolicySets, Sample, populate

# --------------------------------------------------------------------------- phrase tables
DENY_PHRASES = [
    "cannot", "can not", "can't", "may not", "must not", "mustn't", "shall not", "should not",
    "shouldn't", "will not", "won't", "could not", "can never", "may never", "must never",
    "should never", "shall never", "will never", "never", "are not allowed to", "is not allowed to",
    "are not permitted to", "is not permitted to", "are not authorized to", "is not authorized to",
    "are not able to", "is not able to", "are prohibited from", "is prohibited from",
    "are forbidden from", "is forbidden from", "are forbidden to", "is forbidden to",
    "are restricted from", "is restricted from", "are blocked from", "is blocked from",
    "are barred from", "is barred from", "are denied", "is denied", "do not have access to",
    "does not have access to", "are not to", "is not to", "may no longer", "are disallowed from",
    "is disallowed from", "are not granted access to", "is not granted access to",
]
PERMIT_ANY_PHRASES = [
    "have access to", "has access to", "are granted access to", "is granted access to",
    "are given access to", "is given access to", "should have access to", "may have access to",
    "can have access to", "will have access to", "shall have access to", "are entitled to access",
    "is accessible to", "are accessible to", "is available to", "are available to",
]
ONLY_ANY_PHRASES = [
    "is restricted to", "are restricted to", "is limited to", "are limited to", "is reserved for",
    "are reserved for", "is accessible only to", "are accessible only to", "is only accessible to",
    "are only accessible to", "is only available to", "are only available to",
    "is available only to", "are available only to",
]
PERMIT_PHRASES = [
    "should be able to", "must be able to", "shall be able to", "will be able to", "are able to",
    "is able to", "are allowed to", "is allowed to", "are permitted to", "is permitted to",
    "are authorized to", "is authorized to", "are entitled to", "is entitled to", "can", "may", "could",
]
REQUIRE_PHRASES = [
    "are required to", "is required to", "need to", "needs to", "have to", "has to", "must", "shall",
    "should",
]
UNLESS_PHRASES = [
    "unless", "except when", "except if", "except while", "except during", "except in",
    "except where", "except for when", "without", "except on", "except with", "except via",
]
EXCEPT_PHRASES = [
    "with the exception of", "except for", "except", "other than", "but not", "excluding",
    "apart from", "save for", "not including",
]
ONLY_WORDS = {"only", "solely", "exclusively"}
COMMON_ACTIONS = ("read", "write", "create", "delete", "approve", "share", "export")
_CLAUSE_SPLIT = re.compile(
    r",?\s+(?:but|and|while|whereas|yet|although)\s+(?:they\s+|he\s+|she\s+|it\s+)?"
    r"(?=(?:cannot|can't|can\s+not|can|may|must|should|shall|will|never|are\s+not|is\s+not|"
    r"do\s+not|does\s+not)\b)", re.I)
_FIRST_MODAL = re.compile(r"\b(?:cannot|can't|can|may|must|should|shall|will|never|are|is|have|has|do|does)\b",
                          re.I)


def split_clauses(sentence: str) -> list[tuple[str, int]]:
    """Split "X can do A but cannot do B" into two rule clauses, carrying the subject over.

    Returns (clause_text, offset_in_sentence) pairs."""
    parts: list[tuple[str, int]] = []
    last = 0
    for m in _CLAUSE_SPLIT.finditer(sentence):
        parts.append((sentence[last:m.start()], last))
        last = m.end()
    if not parts:
        return [(sentence, 0)]
    parts.append((sentence[last:], last))
    first = parts[0][0]
    mm = _FIRST_MODAL.search(first)
    subject = first[:mm.start()].strip() if mm else ""
    subject = re.sub(r"^(?:only|solely|exclusively)\s+", "", subject, flags=re.I)
    out = [parts[0]]
    for text, off in parts[1:]:
        clause = text.strip()
        if subject:
            clause = f"{subject} {clause}"
        out.append((clause, off))
    return out
CONDITION_LEADS = {"if", "when", "while", "during", "between", "from", "on", "with", "within", "in",
                   "via", "using", "after", "before", "outside", "whenever", "provided", "as"}
FILLER = {"use", "using", "be", "they", "them", "their", "any", "all", "ones", "one", "only", "other",
          "able", "allowed", "permitted", "also", "data", "information", "records", "record", "it",
          "that", "this", "these", "those", "which", "who", "own", "time", "times", "at", "always",
          "ever", "both", "either", "each", "every", "such", "related", "relevant", "certain",
          "specific", "given", "particular", "anything", "something", "everything", "request",
          "requests", "connected", "logged", "signed", "authenticated", "present"}


def _phrases(lst: list[str]) -> list[tuple[str, ...]]:
    return sorted((tuple(p.split()) for p in lst), key=len, reverse=True)


_MARKER_TABLE: list[tuple[str, list[tuple[str, ...]]]] = [
    ("ONLY_ANY", _phrases(ONLY_ANY_PHRASES)),
    ("PERMIT_ANY", _phrases(PERMIT_ANY_PHRASES)),
    ("DENY", _phrases(DENY_PHRASES)),
    ("PERMIT", _phrases(PERMIT_PHRASES)),
    ("REQUIRE", _phrases(REQUIRE_PHRASES)),
    ("UNLESS", _phrases(UNLESS_PHRASES)),
    ("EXCEPT", _phrases(EXCEPT_PHRASES)),
]

# --------------------------------------------------------------------------- regexes
_T = r"(?:\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)|\d{1,2}:\d{2}|noon|midnight)"
_T_LOOSE = r"(?:\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?|noon|midnight)"
RE_RANGE = re.compile(rf"\b(?:between|from)\s+({_T_LOOSE})\s*(?:and|to|until|till|through|-|–)\s*({_T_LOOSE})", re.I)
RE_RANGE_BARE = re.compile(rf"\b({_T})\s*(?:-|–|to|until)\s*({_T})", re.I)
RE_BUSINESS = re.compile(r"\b(?:during|within|in)\s+(?:normal\s+|regular\s+|standard\s+|core\s+)?"
                         r"(?:business|office|working|work)\s+hours\b", re.I)
RE_AFTER_HOURS = re.compile(r"\b(?:outside(?:\s+of)?\s+(?:normal\s+|regular\s+|standard\s+|core\s+)?"
                            r"(?:business|office|working|work)\s+hours|after[- ]hours|off[- ]hours)\b", re.I)
RE_AFTER_BEFORE = re.compile(rf"\b(after|before|past|until)\s+({_T})", re.I)
RE_WEEKEND = re.compile(r"\b(?:on|during|over|at)\s+(?:the\s+)?weekends?\b", re.I)
RE_WEEKDAY = re.compile(r"\b(?:on|during)\s+(?:the\s+)?weekdays?\b|\bmonday\s+(?:through|to|-)\s+friday\b", re.I)
_CMP = (r"over|above|exceeding|exceeds|exceed|greater than|more than|larger than|higher than|"
        r"in excess of|of at least|at least|under|below|less than|lower than|up to|at most|"
        r"no more than|not exceeding|not more than|of up to|of less than|of more than")
RE_NUMERIC = re.compile(rf"\b({_CMP})\s+(\$?\d[\d,]*(?:\.\d+)?\s*[kKmM]?)(?![\w:])", re.I)
RE_OR_HIGHER = re.compile(r"\b((?:security\s+)?clearance(?:\s+level)?|security\s+level|level|tier)\s+(?:of\s+)?"
                          r"(\d+)\s+or\s+(higher|above|greater|more|lower|below|less)\b", re.I)
RE_SAME_ATTR = re.compile(r"\b(?:their|his|her|the\s+same|same|its)\s+(?:own\s+)?"
                          r"(department|team|division|unit|business unit|region|location|organization|"
                          r"organisation|project|tenant|cost center|site)\b", re.I)
RE_OWN = re.compile(r"\b(?:their|his|her)\s+own\b|\b(?:that|which)\s+they\s+(?:own|created|authored|wrote|submitted)\b", re.I)
RE_ASSIGNED = re.compile(r"\b(?:assigned\s+to\s+them|(?:their|his|her)\s+assigned|they\s+are\s+assigned\s+to|"
                         r"under\s+their\s+care)\b", re.I)

CMP_MAP = {
    "over": "gt", "above": "gt", "exceeding": "gt", "exceeds": "gt", "exceed": "gt", "greater than": "gt",
    "more than": "gt", "larger than": "gt", "higher than": "gt", "in excess of": "gt", "of more than": "gt",
    "at least": "gte", "of at least": "gte", "under": "lt", "below": "lt", "less than": "lt",
    "lower than": "lt", "of less than": "lt", "up to": "lte", "at most": "lte", "no more than": "lte",
    "not exceeding": "lte", "not more than": "lte", "of up to": "lte",
}

COMBINING_HINTS = [
    ("deny-unless-permit", re.compile(
        r"\b(?:den(?:y|ied)|block(?:ed)?|forbidden|prohibited)\s+by\s+default\b|\bdefault[- ]deny\b|"
        r"\ball\s+other\s+(?:access|requests?|actions?|operations?)\s+(?:is|are|will be|shall be|must be|should be)\s+"
        r"(?:denied|blocked|prohibited|forbidden|rejected)\b|"
        r"\b(?:anything|everything)\s+(?:else|not explicitly (?:permitted|allowed))\s+(?:is|will be)\s+"
        r"(?:denied|prohibited|forbidden)\b|\bdeny\s+all\s+other\b", re.I)),
    ("permit-unless-deny", re.compile(
        r"\b(?:allow(?:ed)?|permit(?:ted)?)\s+by\s+default\b|\bdefault[- ](?:allow|permit)\b|"
        r"\ball\s+other\s+(?:access|requests?|actions?)\s+(?:is|are)\s+(?:allowed|permitted)\b", re.I)),
    ("first-applicable", re.compile(
        r"\bfirst\s+(?:matching|applicable)\s+rule\b|\brules?\s+(?:are\s+)?(?:evaluated|applied)\s+in\s+order\b", re.I)),
    ("permit-overrides", re.compile(
        r"\b(?:permits?|allows?|permissions?|grants?)\s+(?:always\s+)?(?:take|takes)\s+precedence\b|\bpermit[- ]overrides\b", re.I)),
    ("deny-overrides", re.compile(
        r"\b(?:denials?|denies|deny\s+rules?|prohibitions?)\s+(?:always\s+)?(?:take|takes)\s+precedence\b|"
        r"\bdeny[- ]overrides\b", re.I)),
]


def detect_combining(text: str) -> tuple[str | None, list[str]]:
    """Identify combining-algorithm characteristics of the text (FIG. 4)."""
    found = None
    evidence = []
    for name, rx in COMBINING_HINTS:
        m = rx.search(text)
        if m:
            evidence.append(m.group(0))
            found = found or name
    return found, evidence


def parse_hour(s: str) -> int:
    s = s.strip().lower().replace(".", "")
    if s == "noon":
        return 12
    if s == "midnight":
        return 0
    m = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", s)
    if not m:
        raise ValueError(s)
    h = int(m.group(1))
    mer = m.group(3)
    if mer == "pm" and h < 12:
        h += 12
    if mer == "am" and h == 12:
        h = 0
    return h % 24


def parse_number(s: str) -> int | float:
    s = s.replace("$", "").replace(",", "").strip().lower()
    mult = 1
    if s.endswith("k"):
        mult, s = 1000, s[:-1]
    elif s.endswith("m"):
        mult, s = 1_000_000, s[:-1]
    v = float(s) * mult
    return int(v) if v.is_integer() else v


def _base_verb(word: str) -> str:
    """Past participle -> base form for learned verbs ('reassigned' -> 'reassign')."""
    if word.endswith("ied"):
        return word[:-3] + "y"
    if word.endswith("ed") and len(word) > 4:
        stem_ = word[:-2]
        if stem_.endswith(("at", "iz", "is", "v", "c", "u", "g")):
            return stem_ + "e"
        if len(stem_) > 2 and stem_[-1] == stem_[-2]:
            return stem_[:-1]
        return stem_
    return word


# --------------------------------------------------------------------------- data classes
@dataclass
class Entity:
    start: int
    end: int
    text: str
    kind: str  # value | action | universal | nobody | attribute | placeholder | learned
    attribute: AttributeDef | None = None
    value: str | None = None
    negated: bool = False
    conditions: list[Condition] = field(default_factory=list)
    category: str | None = None
    char_start: int = 0
    char_end: int = 0
    salience: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text, "kind": self.kind, "category": self.category,
            "attribute": self.attribute.id if self.attribute else None, "value": self.value,
            "negated": self.negated, "start": self.char_start, "end": self.char_end,
            "salience": round(self.salience, 2),
            "conditions": [c.describe() for c in self.conditions],
        }


@dataclass
class Marker:
    kind: str
    start: int
    end: int
    text: str


@dataclass
class SentenceAnalysis:
    text: str
    offset: int
    tokens: list[nlp.Token] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)
    groups: dict[str, list[Condition]] = field(default_factory=dict)
    actions: list[str] = field(default_factory=list)
    signature: list[str] = field(default_factory=list)
    coverage: float = 1.0
    candidates: list[Candidate] = field(default_factory=list)
    learned: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    is_rule: bool = True
    meta: str | None = None

    def confidence(self, cand: Candidate) -> float:
        c = cand.score * (0.6 + 0.4 * self.coverage) - 0.12 * len(self.learned)
        return round(max(0.0, min(1.0, c)), 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text, "offset": self.offset, "is_rule": self.is_rule, "meta": self.meta,
            "signature": self.signature, "coverage": round(self.coverage, 2),
            "tokens": [{"text": t.text, "pos": t.pos, "lemma": t.stem, "morphemes": t.morphemes}
                       for t in self.tokens],
            "entities": [e.to_dict() for e in self.entities],
            "markers": [{"kind": m.kind, "text": m.text} for m in self.markers],
            "candidates": [{"sample": c.id, "score": c.score, "confidence": self.confidence(c),
                            "example": c.sample.example} for c in self.candidates[:3]],
            "learned": self.learned, "notes": self.notes,
        }


# --------------------------------------------------------------------------- engine
class RulesEngine:
    """Deterministic NLP + sample-rule matching engine. No network, no API key."""

    name = "rules"

    def __init__(self, dictionary: AttributeDictionary, policy_sets: PolicySets, learn: bool = True) -> None:
        self.dictionary = dictionary
        self.policy_sets = policy_sets
        self.learn = learn
        self.policy_type = "general"

    # ---------------------------------------------------------------- pre-processing
    def _preprocess(self, text: str) -> tuple[str, dict[str, list[Condition]], list[tuple[int, int, int, int]]]:
        """Replace time windows, numeric limits and relations with placeholder tokens."""
        placeholders: dict[str, list[Condition]] = {}
        spans: list[tuple[int, int, list[Condition], str]] = []  # (start, end, conds, kind)
        hour = AttributeRef(category="environment", id="hour", datatype="integer")
        day = AttributeRef(category="environment", id="day_of_week")

        def claim(m: re.Match[str], conds: list[Condition], kind: str = "TIME", group: int = 0) -> None:
            s, e = m.span(group)
            if any(not (e <= a or s >= b) for a, b, _, _ in spans):
                return
            spans.append((s, e, conds, kind))

        for m in RE_RANGE.finditer(text):
            try:
                a, b = parse_hour(m.group(1)), parse_hour(m.group(2))
            except ValueError:
                continue
            claim(m, self._hour_window(hour, a, b))
        for m in RE_RANGE_BARE.finditer(text):
            a, b = parse_hour(m.group(1)), parse_hour(m.group(2))
            claim(m, self._hour_window(hour, a, b))
        for m in RE_AFTER_HOURS.finditer(text):
            claim(m, [Condition(attribute=hour, operator=Operator.NOT_IN, value=list(range(9, 17)))])
        for m in RE_BUSINESS.finditer(text):
            claim(m, self._hour_window(hour, 9, 17))
        for m in RE_AFTER_BEFORE.finditer(text):
            h = parse_hour(m.group(2))
            op = Operator.GTE if m.group(1).lower() in ("after", "past") else Operator.LT
            claim(m, [Condition(attribute=hour, operator=op, value=h)])
        for m in RE_WEEKEND.finditer(text):
            claim(m, [Condition(attribute=day, operator=Operator.IN, value=["sat", "sun"])])
        for m in RE_WEEKDAY.finditer(text):
            claim(m, [Condition(attribute=day, operator=Operator.IN, value=["mon", "tue", "wed", "thu", "fri"])])
        for m in RE_OR_HIGHER.finditer(text):
            ref = self._numeric_attr(m.group(1), text)
            op = Operator.GTE if m.group(3).lower() in ("higher", "above", "greater", "more") else Operator.LTE
            claim(m, [Condition(attribute=ref, operator=op, value=int(m.group(2)))], "NUM")
        for m in RE_NUMERIC.finditer(text):
            before = text[max(0, m.start() - 30):m.start()]
            ref = self._numeric_attr(before, text, money="$" in m.group(2))
            op = Operator(CMP_MAP[re.sub(r"\s+", " ", m.group(1).lower())])
            claim(m, [Condition(attribute=ref, operator=op, value=parse_number(m.group(2)))], "NUM")
        for m in RE_SAME_ATTR.finditer(text):
            attr = self._attr_by_alias(m.group(1))
            sref = AttributeRef(category="subject", id=attr)
            rref = AttributeRef(category="resource", id=attr)
            claim(m, [Condition(attribute=rref, operator=Operator.EQ, value=sref)], "REL")
        for m in RE_ASSIGNED.finditer(text):
            claim(m, [Condition(attribute=AttributeRef(category="resource", id="assignee"),
                                operator=Operator.EQ, value=AttributeRef(category="subject", id="id"))], "REL")
        for m in RE_OWN.finditer(text):
            claim(m, [Condition(attribute=AttributeRef(category="resource", id="owner"),
                                operator=Operator.EQ, value=AttributeRef(category="subject", id="id"))], "REL")

        spans.sort()
        out, last, mapping = [], 0, []
        new_pos = 0
        for i, (s, e, conds, kind) in enumerate(spans):
            out.append(text[last:s])
            new_pos += s - last
            ph = f" __{kind}{i}__ "
            placeholders[ph.strip()] = conds
            mapping.append((s, e, new_pos, new_pos + len(ph)))
            out.append(ph)
            new_pos += len(ph)
            last = e
        out.append(text[last:])
        return "".join(out), placeholders, mapping

    @staticmethod
    def _hour_window(hour: AttributeRef, a: int, b: int) -> list[Condition]:
        if b > a:
            return [Condition(attribute=hour, operator=Operator.GTE, value=a),
                    Condition(attribute=hour, operator=Operator.LT, value=b)]
        # Overnight window, e.g. 10pm-6am: hour NOT in [b, a)
        return [Condition(attribute=hour, operator=Operator.NOT_IN, value=list(range(b, a)))]

    def _numeric_attr(self, context: str, sentence: str, money: bool = False) -> AttributeRef:
        ints = [a for a in self.dictionary.attributes.values() if a.datatype in ("integer", "double")]
        ctx = {t.stem for t in nlp.tokenize(context)}
        for a in ints:
            for alias in [a.id, *a.aliases]:
                if set(self.dictionary.key(alias)) <= ctx and self.dictionary.key(alias):
                    return a.ref()
        if money and "amount" in self.dictionary.attributes:
            return self.dictionary.attributes["amount"].ref()
        sent = {t.stem for t in nlp.tokenize(sentence)}
        for a in ints:
            for alias in [a.id, *a.aliases]:
                k = self.dictionary.key(alias)
                if k and set(k) <= sent and a.categories[0] != "environment":
                    return a.ref()
        return AttributeRef(category="resource", id="amount", datatype="integer")

    def _attr_by_alias(self, word: str) -> str:
        k = self.dictionary.key(word)
        for a in self.dictionary.attributes.values():
            if any(self.dictionary.key(al) == k for al in [a.id, *a.aliases]):
                return a.id
        return nlp.singular(word.lower()).replace(" ", "_")

    # ---------------------------------------------------------------- analysis
    def analyze(self, sentence: str, offset: int = 0) -> SentenceAnalysis:
        an = SentenceAnalysis(text=sentence, offset=offset)
        hint, _ = detect_combining(sentence)
        if hint:
            an.is_rule, an.meta = False, hint
            an.notes.append(f"Sentence states a combining rule: {hint}")
            return an

        text, placeholders, mapping = self._preprocess(sentence)
        tokens = nlp.tokenize(text)
        nlp.pos_tag(tokens, self.dictionary.known_verb_stems(), self.dictionary.known_noun_stems())
        an.tokens = tokens

        def orig(pos: int) -> int:
            shift = 0
            for s, e, ns, ne in mapping:
                if pos >= ne:
                    shift += (e - s) - (ne - ns)
                elif pos >= ns:
                    return s if pos == ns else e
            return pos + shift

        consumed = [False] * len(tokens)
        lowers = [t.lower for t in tokens]

        # ---- markers (modal phrases, unless/except, only)
        i = 0
        while i < len(tokens):
            hit = None
            for kind, phrases in _MARKER_TABLE:
                for p in phrases:
                    if tuple(lowers[i:i + len(p)]) == p:
                        hit = (kind, len(p))
                        break
                if hit:
                    break
            if hit:
                kind, L = hit
                an.markers.append(Marker(kind, i, i + L, " ".join(t.text for t in tokens[i:i + L])))
                for j in range(i, i + L):
                    consumed[j] = True
                i += L
                continue
            if lowers[i] in ONLY_WORDS:
                an.markers.append(Marker("ONLY", i, i + 1, tokens[i].text))
                consumed[i] = True
            i += 1

        # Only the first modal marker is the rule's modality; demote later ones.
        modal_kinds = ("DENY", "PERMIT", "REQUIRE", "PERMIT_ANY", "ONLY_ANY")
        modal = next((m for m in an.markers if m.kind in modal_kinds), None)
        if modal is None:
            an.is_rule = False
            an.notes.append("No modal verb (can/may/must/cannot...) found: not an access rule.")
            return an

        # ---- named entity recognition
        an.entities = self._ner(tokens, consumed, placeholders)
        for e in an.entities:
            e.char_start = orig(tokens[e.start].start) + offset
            e.char_end = orig(tokens[e.end - 1].end) + offset

        # ---- regions: main | unless | except | require_cond
        region = ["main"] * len(tokens)
        cur = "main"
        req_end = None
        if modal.kind == "REQUIRE":
            # tokens between REQUIRE and "to <action>" describe the requirement
            for j in range(modal.end, len(tokens)):
                if lowers[j] == "to" and any(e.kind == "action" and e.start in (j + 1, j + 2) for e in an.entities):
                    req_end = j
                    break
        for idx in range(len(tokens)):
            m = next((mk for mk in an.markers if mk.start == idx), None)
            if m and m.kind == "UNLESS":
                cur = "unless"
            elif m and m.kind == "EXCEPT":
                nxt = next((e for e in an.entities if e.start >= m.end), None)
                cur = "unless" if nxt and (nxt.kind == "placeholder" or nxt.category == "environment") else "except"
                if cur == "unless":
                    m.kind = "UNLESS"
            if req_end is not None and modal.end <= idx < req_end and cur == "main":
                region[idx] = "require_cond"
            else:
                region[idx] = cur

        # ---- slot assignment
        groups: dict[str, list[Condition]] = {g: [] for g in ("S", "R", "C", "U", "X")}
        actions: list[str] = []
        flags = {"ALL": False, "NOBODY": False, "NO_SUBJECT_FLIP": False}
        head = None
        for e in an.entities:
            reg = region[e.start]
            before_modal = e.start < modal.start
            if e.kind == "action":
                if reg in ("main",) and not before_modal:
                    if e.value not in actions:
                        actions.append(e.value or "*")
                continue
            if e.kind == "universal":
                flags["ALL"] = True
                head = "subject"
                continue
            if e.kind == "nobody":
                flags["NOBODY"] = True
                head = "subject"
                continue
            if e.kind == "attribute":
                continue
            if e.kind == "pronoun":
                head = "subject"
                continue
            conds: list[Condition]
            if e.kind == "placeholder":
                conds = e.conditions
                target = "C"
            else:
                assert e.attribute is not None
                cats = e.attribute.categories
                after_by = any(lowers[k] == "by" for k in range(modal.end, e.start))
                if len(cats) == 1:
                    cat = cats[0]
                elif "subject" in cats and not before_modal and (modal.kind == "ONLY_ANY" or after_by):
                    cat = "subject"
                elif head in cats:
                    cat = head
                elif before_modal and "subject" in cats:
                    cat = "subject"
                else:
                    cat = "resource" if "resource" in cats else cats[0]
                e.category = cat
                # "records of patients" / "files for contractors": a subject noun qualifying
                # the resource (not the requester) -- skip it.
                if cat == "subject" and not before_modal and e.start > 0 and                         lowers[e.start - 1] in ("of", "for", "about", "regarding", "concerning") and                         not any(t.lower == "by" for t in tokens[modal.end:e.start]):
                    e.kind = "qualifier"
                    continue
                if len(cats) == 1 and cat in ("subject", "resource"):
                    head = cat
                conds = [self._condition(e, cat)]
                if e.attribute.datatype == "boolean" or cat == "environment":
                    target = "C"
                else:
                    target = {"subject": "S", "resource": "R"}.get(cat, "C")
                # "No contractor may ..." => deny for contractors
                if cat == "subject" and e.start > 0 and lowers[e.start - 1] == "no":
                    flags["NO_SUBJECT_FLIP"] = True
            if reg == "unless":
                target = "U"
            elif reg == "except":
                target = "X"
            elif reg == "require_cond":
                target = "C"
            for c in conds:
                self._add(groups[target], c)

        # ---- ONLY classification
        only_kinds = set()
        for m in an.markers:
            if m.kind != "ONLY":
                continue
            j = m.end
            while j < len(tokens) and tokens[j].pos == "DET":
                j += 1
            ent = next((e for e in an.entities if e.start <= j < e.end), None)
            nxt_low = lowers[j] if j < len(tokens) else ""
            if ent and (ent.category == "subject" or ent.kind in ("universal",)) and not \
                    (ent.attribute and ent.attribute.datatype == "boolean"):
                kind = "ONLY_SUBJ"
            elif nxt_low == "be":
                by_subject = "by" in lowers[j:]
                kind = "ONLY_SUBJ" if by_subject else "ONLY_ACTION"
            elif ent and ent.kind == "action":
                kind = "ONLY_ACTION"
            elif nxt_low in CONDITION_LEADS or (ent and (ent.kind == "placeholder" or ent.category == "environment")):
                kind = "ONLY_COND"
            elif ent and ent.attribute and ent.attribute.datatype == "boolean":
                kind = "ONLY_COND"
            else:
                kind = "ONLY_SUBJ" if m.start < modal.start else "ONLY_COND"
            m.kind = kind
            only_kinds.add(kind)
        if modal.kind == "ONLY_ANY":
            only_kinds.add("ONLY_SUBJ")

        # ---- learn unknown candidate attributes (claim 1)
        covered = [False] * len(tokens)
        for e in an.entities:
            for j in range(e.start, e.end):
                covered[j] = True
        for j, c in enumerate(consumed):
            covered[j] = covered[j] or c
        content = [j for j, t in enumerate(tokens)
                   if t.is_word and t.pos in ("NOUN", "VERB", "ADJ") and t.lower not in FILLER
                   and not t.text.startswith("__")]
        learned_any = self._learn_unknowns(an, tokens, covered, content, modal, groups, actions, flags, region)
        if learned_any:
            for e in an.entities:
                for j in range(e.start, e.end):
                    covered[j] = True
        uncovered = [tokens[j].text for j in content if not covered[j]]
        an.coverage = 1.0 if not content else sum(1 for j in content if covered[j]) / len(content)
        if uncovered:
            an.notes.append("Unrecognised words: " + ", ".join(uncovered))

        # ---- polarity
        polarity = {"DENY": "DENY", "PERMIT": "PERMIT", "PERMIT_ANY": "PERMIT", "ONLY_ANY": "PERMIT",
                    "REQUIRE": "REQUIRE"}[modal.kind]
        if flags["NO_SUBJECT_FLIP"] and polarity == "PERMIT":
            polarity = "DENY"
        if modal.kind == "REQUIRE" and req_end is None:
            polarity = "PERMIT"
            an.notes.append("'must' without 'to <action>' treated as a permission.")
        if not actions:
            actions = ["*"]

        # ---- signature (canonical order)
        sig: list[str] = []
        if "ONLY_SUBJ" in only_kinds:
            sig.append("ONLY_SUBJ")
        if flags["NOBODY"] and not flags["NO_SUBJECT_FLIP"]:
            sig.append("NOBODY")
        elif groups["S"]:
            sig.append("SUBJECT")
        else:
            sig.append("ALL")
        sig.append(polarity)
        if polarity == "REQUIRE" and groups["C"]:
            sig.append("COND")
        if "ONLY_ACTION" in only_kinds:
            sig.append("ONLY_ACTION")
        sig.append("ACTION")
        if groups["R"]:
            sig.append("RESOURCE")
        if "ONLY_COND" in only_kinds:
            sig.append("ONLY_COND")
        if groups["C"] and polarity != "REQUIRE":
            sig.append("COND")
        if groups["U"]:
            sig.append("UNLESS")
        if groups["X"]:
            sig.append("EXCEPT")
        an.signature = sig
        an.groups = groups
        an.actions = actions
        return an

    def candidates(self, an: SentenceAnalysis, samples: list[Sample]) -> list[Candidate]:
        an.candidates = self.policy_sets.match(an.signature, samples) if an.is_rule else []
        return an.candidates

    def rules_for(self, an: SentenceAnalysis, cand: Candidate) -> list[Rule]:
        other = [a for a in self.dictionary.action_names()
                 if a not in an.actions and (a in COMMON_ACTIONS or
                                             self.dictionary.action_domains.get(a) == self.policy_type)]             if "*" not in an.actions else []
        rules = populate(cand.sample, an.groups, an.actions, other)
        conf = an.confidence(cand)
        for r in rules:
            r.confidence = conf
            r.source_text = an.text
            r.engine = self.name
        return rules

    # ---------------------------------------------------------------- helpers
    def _ner(self, tokens: list[nlp.Token], consumed: list[bool],
             placeholders: dict[str, list[Condition]]) -> list[Entity]:
        ents: list[Entity] = []
        maxlen = self.dictionary.max_phrase_len
        n = len(tokens)
        i = 0
        while i < n:
            tok = tokens[i]
            if tok.text in placeholders:
                ents.append(Entity(i, i + 1, tok.text, "placeholder", conditions=placeholders[tok.text],
                                   salience=1.0))
                i += 1
                continue
            if consumed[i] or not tok.is_word:
                i += 1
                continue
            if tok.lower in ("they", "he", "she", "them"):
                ents.append(Entity(i, i + 1, tok.text, "pronoun"))
                i += 1
                continue
            found = False
            for L in range(min(maxlen + 2, n - i), 0, -1):
                span = tokens[i:i + L]
                if any(consumed[i + k] for k in range(L)) or not span[-1].is_word:
                    continue
                if any(t.text in placeholders or t.text in (",", ";", ".", ":") for t in span):
                    continue
                key = tuple(t.stem for t in span if t.is_word or t.lower[:1].isdigit())
                mentions = self.dictionary.lookup(key)
                if not mentions:
                    continue
                prev_low = tokens[i - 1].lower if i else ""
                verb_slot = (i > 0 and consumed[i - 1]) or prev_low in ("to", "and", "or", "not", "only", "never")
                pick = mentions[0]
                if verb_slot:
                    pick = next((m for m in mentions if m.kind == "action"), pick)
                text = " ".join(t.text for t in span)
                ents.append(self._entity_from_mention(pick, i, i + L, text, L))
                i += L
                found = True
                break
            if found:
                continue
            neg = nlp.negating_prefix(tok.text)
            if neg:
                _, rest = neg
                for L in range(min(3, n - i), 0, -1):
                    key = (nlp.stem(rest), *[t.stem for t in tokens[i + 1:i + L]])
                    mentions = [m for m in self.dictionary.lookup(key) if m.kind == "value"]
                    if mentions:
                        text = " ".join(t.text for t in tokens[i:i + L])
                        e = self._entity_from_mention(mentions[0], i, i + L, text, L)
                        e.negated = True
                        ents.append(e)
                        i += L
                        found = True
                        break
            if not found:
                i += 1
        return ents

    @staticmethod
    def _entity_from_mention(m: Any, start: int, end: int, text: str, length: int) -> Entity:
        sal = min(1.0, 0.5 + 0.15 * length)
        if m.kind == "value":
            cat = m.attribute.categories[0] if len(m.attribute.categories) == 1 else None
            return Entity(start, end, text, "value", m.attribute, m.value, category=cat, salience=sal)
        if m.kind == "action":
            return Entity(start, end, text, "action", value=m.value, category="action", salience=sal)
        if m.kind in ("universal", "nobody"):
            return Entity(start, end, text, m.kind, category="subject", salience=sal)
        return Entity(start, end, text, "attribute", m.attribute, salience=0.3)

    @staticmethod
    def _condition(e: Entity, category: str) -> Condition:
        attr = e.attribute
        assert attr is not None
        ref = AttributeRef(category=category, id=attr.id, datatype=attr.datatype)  # type: ignore[arg-type]
        value: Any = e.value
        if attr.datatype == "boolean":
            value = str(e.value).lower() in ("true", "yes", "1")
        elif attr.datatype == "integer":
            try:
                value = int(str(e.value))
            except ValueError:
                pass
        return Condition(attribute=ref, operator=Operator.NEQ if e.negated else Operator.EQ, value=value)

    @staticmethod
    def _add(group: list[Condition], c: Condition) -> None:
        """Add to a group; same-attribute equalities within a group are OR-ed into IN."""
        for i, ex in enumerate(group):
            if ex.attribute.path != c.attribute.path or ex.is_reference or c.is_reference:
                continue
            if ex.operator in (Operator.EQ, Operator.IN) and c.operator == Operator.EQ:
                vals = ex.values + [v for v in c.values if v not in ex.values]
                group[i] = Condition(attribute=ex.attribute, operator=Operator.IN, value=vals) \
                    if len(vals) > 1 else ex
                return
            if ex.model_dump() == c.model_dump():
                return
        group.append(c)

    def _learn_unknowns(self, an: SentenceAnalysis, tokens: list[nlp.Token], covered: list[bool],
                        content: list[int], modal: Marker, groups: dict[str, list[Condition]],
                        actions: list[str], flags: dict[str, bool], region: list[str]) -> bool:
        """Discover candidate attributes for unknown nouns/verbs (claim 1) and store them."""
        learned = False
        unknown = [j for j in content if not covered[j]]
        if not unknown:
            return False

        def span_at(j: int) -> list[int]:
            # contiguous run of uncovered NOUN/ADJ tokens ending at a noun
            run = [j]
            k = j + 1
            while k < len(tokens) and k in unknown and tokens[k].pos in ("NOUN", "ADJ"):
                run.append(k)
                k += 1
            return run

        has_subject = bool(groups["S"]) or flags["ALL"] or flags["NOBODY"]
        passive_by = next((j for j in range(modal.end, len(tokens)) if tokens[j].lower == "by"), None)
        # Unknown action: first uncovered word right after the modal (skipping only/not/be)
        if actions == []:
            j = modal.end
            while j < len(tokens) and (tokens[j].lower in ("only", "not", "never", "be", "also") or
                                       tokens[j].pos in ("ADV",)):
                j += 1
            if j < len(tokens) and j in unknown and tokens[j].pos in ("VERB", "NOUN", "ADJ"):
                verb = tokens[j].lower
                if passive_by is not None:
                    verb = _base_verb(verb)
                self._record(an, "action", None, verb, tokens[j].text)
                actions.append(verb)
                an.entities.append(Entity(j, j + 1, tokens[j].text, "action", value=verb, category="action",
                                          char_start=tokens[j].start + an.offset, char_end=tokens[j].end + an.offset))
                covered[j] = True
                learned = True
        subj_zone = [j for j in unknown if (j < modal.start and passive_by is None) or
                     (passive_by is not None and j > passive_by)]
        res_zone = [j for j in unknown if j > modal.end and region[j] == "main" and
                    (passive_by is None or j < passive_by)] if passive_by is None else \
            [j for j in unknown if j < modal.start]
        if not has_subject and subj_zone:
            run = span_at(subj_zone[0])
            learned |= self._learn_value(an, tokens, run, "role", "subject", groups["S"], covered)
        if not groups["R"] and res_zone:
            nouns = [j for j in res_zone if tokens[j].pos in ("NOUN", "ADJ") and not covered[j]]
            if nouns:
                run = span_at(nouns[0])
                learned |= self._learn_value(an, tokens, run, "type", "resource", groups["R"], covered)
        return learned

    def _learn_value(self, an: SentenceAnalysis, tokens: list[nlp.Token], run: list[int], attr_id: str,
                     category: str, group: list[Condition], covered: list[bool]) -> bool:
        words = [tokens[j].text.lower() for j in run]
        if not words:
            return False
        words[-1] = nlp.singular(words[-1])
        value = "_".join(words)
        surface = " ".join(words)
        attr = self.dictionary.get(attr_id)
        ref = AttributeRef(category=category, id=attr_id, datatype=attr.datatype if attr else "string")  # type: ignore[arg-type]
        group.append(Condition(attribute=ref, operator=Operator.EQ, value=value))
        self._record(an, "value", f"{category}.{attr_id}", value, surface, attr_id, category)
        e = Entity(run[0], run[-1] + 1, " ".join(tokens[j].text for j in run), "learned",
                   attr, value, category=category, salience=0.4,
                   char_start=tokens[run[0]].start + an.offset, char_end=tokens[run[-1]].end + an.offset)
        an.entities.append(e)
        for j in run:
            covered[j] = True
        return True

    @staticmethod
    def _record(an: SentenceAnalysis, kind: str, attribute: str | None, value: str, surface: str,
                attr_id: str | None = None, category: str | None = None) -> None:
        an.learned.append({"kind": kind, "attribute": attribute, "value": value, "surface": surface,
                           "attr_id": attr_id, "category": category, "source": "nlp"})
        an.notes.append(f"Learned new candidate {kind}: {attribute + '=' if attribute else ''}{value} "
                        f"(from '{surface}')")
