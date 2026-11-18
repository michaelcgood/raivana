"""Lightweight, dependency-free NLP for access control policy text.

Implements the natural language processing steps described in US 10,606,957 B1
(FIG. 5, step 502; claims 6-7, 13-14, 20): tokenization, lemmatization/stemming,
part-of-speech tagging, morphological segmentation, and language identification.

It is intentionally small and deterministic: it only needs to be good at the
restricted register of policy language ("Nurses may view patient records during
business hours"), and it ships with zero heavy dependencies so the offline
engine works anywhere. Messier text is handled by the Claude engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

TOKEN_RE = re.compile(
    r"__[A-Z]+\d+__"                      # placeholders injected by pre-processing
    r"|\$?\d[\d,]*(?:\.\d+)?%?"           # numbers / money
    r"|[A-Za-z]+(?:[-'][A-Za-z]+)*"       # words, hyphenated words, contractions
    r"|[^\sA-Za-z\d]"                     # punctuation
)

IRREGULAR = {
    "is": "be", "are": "be", "was": "be", "were": "be", "been": "be", "being": "be", "am": "be",
    "has": "have", "had": "have", "having": "have",
    "does": "do", "did": "do", "done": "do",
    "children": "child", "people": "person", "men": "man", "women": "woman",
    "data": "data", "media": "media", "criteria": "criterion", "analyses": "analysis",
    "written": "write", "wrote": "write", "read": "read", "seen": "see", "saw": "see",
    "made": "make", "given": "give", "gave": "give", "taken": "take", "took": "take",
    "chosen": "choose", "sent": "send", "shared": "share", "can't": "cannot",
    "staff": "staff", "personnel": "personnel", "status": "status", "access": "access",
    "process": "process", "business": "business", "address": "address", "class": "class",
    "pass": "pass", "less": "less", "always": "always", "sometimes": "sometimes",
    "nobody": "nobody", "everyone": "everyone", "anyone": "anyone", "this": "this", "its": "its",
}

DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "any", "all", "every", "each",
               "some", "no", "their", "his", "her", "its", "our", "your", "my", "own"}
PREPOSITIONS = {"in", "on", "at", "to", "from", "by", "with", "without", "for", "of", "during",
                "between", "within", "outside", "inside", "after", "before", "over", "under",
                "above", "below", "via", "through", "into", "about", "across", "except", "per"}
CONJUNCTIONS = {"and", "or", "but", "nor", "unless", "if", "when", "while", "whereas", "although",
                "because", "provided", "whenever", "where", "than"}
MODALS = {"can", "cannot", "could", "may", "might", "must", "shall", "should", "will", "would",
          "can't", "won't", "mustn't", "shouldn't", "shan't"}
AUXILIARIES = {"be", "is", "are", "was", "were", "been", "being", "am", "have", "has", "had",
               "do", "does", "did"}
PRONOUNS = {"they", "them", "he", "she", "it", "we", "you", "i", "who", "whom", "which", "that"}
NEGATIONS = {"not", "never", "no", "cannot", "can't", "won't", "nobody", "none", "neither"}
ADVERBS = {"only", "also", "always", "never", "just", "solely", "exclusively", "otherwise",
           "additionally", "however", "even", "still", "directly", "explicitly"}

NEGATING_PREFIXES = ("non", "un", "dis", "in", "im", "ir", "il", "mis")
PREFIXES = ("non", "un", "dis", "mis", "pre", "re", "de", "over", "under", "sub", "super", "anti",
            "auto", "multi", "inter", "co", "in", "im", "ir", "il")
SUFFIXES = ("ization", "ation", "ition", "tion", "sion", "ment", "ness", "ity", "able", "ible",
            "ize", "ise", "ify", "ing", "ers", "ors", "er", "or", "ist", "ed", "es", "s", "al",
            "ly", "ful", "less", "ous", "ive", "ance", "ence")

NOUN_SUFFIXES = ("tion", "sion", "ment", "ness", "ity", "ance", "ence", "ship", "ist", "er", "or",
                 "ee", "ant", "ent", "age", "ure", "log", "logs", "ics", "ism")
VERB_SUFFIXES = ("ize", "ise", "ify", "ate")
ADJ_SUFFIXES = ("al", "ous", "ive", "able", "ible", "ful", "less", "ic", "ary", "ial")

ENGLISH_STOPWORDS = {"the", "and", "of", "to", "a", "in", "is", "can", "may", "be", "for", "not",
                     "on", "with", "only", "by", "are", "must", "all", "or", "their"}
LANGUAGE_STOPWORDS = {
    "en": ENGLISH_STOPWORDS,
    "es": {"el", "la", "los", "las", "de", "y", "puede", "pueden", "no", "solo", "en", "con", "por",
           "para", "del", "un", "una", "que"},
    "fr": {"le", "la", "les", "de", "et", "peut", "peuvent", "ne", "pas", "seulement", "en", "avec",
           "par", "pour", "des", "un", "une", "que"},
    "de": {"der", "die", "das", "und", "kann", "können", "nicht", "nur", "mit", "von", "für", "auf",
           "den", "dem", "ein", "eine", "dürfen", "darf"},
    "pt": {"o", "a", "os", "as", "de", "e", "pode", "podem", "não", "apenas", "em", "com", "por",
           "para", "do", "da", "um", "uma"},
}


@dataclass
class Token:
    text: str
    start: int
    end: int
    lower: str
    stem: str
    pos: str = "X"
    morphemes: list[str] = field(default_factory=list)

    @property
    def is_word(self) -> bool:
        return self.lower[:1].isalpha()


def stem(word: str) -> str:
    """Normalise a word to a matching key. Both dictionary surface forms and
    policy text pass through this, so consistency matters more than linguistic purity."""
    w = word.lower().strip("'")
    if w in IRREGULAR:
        w = IRREGULAR[w]
    elif len(w) > 3:
        if w.endswith("ies") and len(w) > 4:
            w = w[:-3] + "y"
        elif w.endswith("sses"):
            w = w[:-2]
        elif w.endswith(("ches", "shes", "xes", "zes")):
            w = w[:-2]
        elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
            w = w[:-1]
        if w.endswith("ied") and len(w) > 4:
            w = w[:-3] + "y"
        elif w.endswith("ing") and len(w) > 5:
            w = _undouble(w[:-3])
        elif w.endswith("ed") and len(w) > 4:
            w = _undouble(w[:-2])
    if w.endswith("e") and len(w) > 3:
        w = w[:-1]
    return w


def _undouble(w: str) -> str:
    if len(w) > 2 and w[-1] == w[-2] and w[-1] not in "lsz":
        return w[:-1]
    return w


def singular(word: str) -> str:
    """Human-readable singular form (used for naming newly learned attribute values)."""
    w = word.lower()
    if w in IRREGULAR and IRREGULAR[w] not in ("be", "have", "do"):
        return IRREGULAR[w]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("sses", "ches", "shes", "xes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
        return w[:-1]
    return w


def segment(word: str) -> list[str]:
    """Morphological segmentation: split a word into prefix / root / suffix morphemes.

    >>> segment("non-employees")
    ['non-', 'employe', '-es']

    The segmentation is used, among other things, to recognise negated attribute
    mentions such as "unverified devices" or "non-employees".
    """
    w = word.lower()
    morphs: list[str] = []
    if "-" in w:
        head, _, rest = w.partition("-")
        if head in PREFIXES:
            return [head + "-", *segment(rest)]
    for p in sorted(PREFIXES, key=len, reverse=True):
        if w.startswith(p) and len(w) - len(p) >= 5:
            morphs.append(p + "-")
            w = w[len(p):]
            break
    tail: list[str] = []
    for _ in range(3):
        for s in SUFFIXES:
            if w.endswith(s) and len(w) - len(s) >= 4:
                tail.insert(0, "-" + s)
                w = w[: -len(s)]
                break
        else:
            break
    return [*morphs, w, *tail]


def negating_prefix(word: str) -> tuple[str, str] | None:
    """If `word` carries a negating prefix (un-, non-, dis-...), return (prefix, remainder)."""
    w = word.lower()
    if "-" in w:
        head, _, rest = w.partition("-")
        if head in ("non", "un", "dis") and rest:
            return head, rest
    for p in NEGATING_PREFIXES:
        if w.startswith(p) and len(w) - len(p) >= 4:
            return p, w[len(p):]
    return None


def tokenize(text: str) -> list[Token]:
    tokens = []
    for m in TOKEN_RE.finditer(text):
        t = m.group(0)
        tokens.append(Token(text=t, start=m.start(), end=m.end(), lower=t.lower(), stem=stem(t)))
    return tokens


def pos_tag(tokens: list[Token], known_verbs: set[str] | None = None,
            known_nouns: set[str] | None = None) -> list[Token]:
    """Rule-based part-of-speech tagger (universal tagset)."""
    known_verbs = known_verbs or set()
    known_nouns = known_nouns or set()
    prev = ""
    for i, tok in enumerate(tokens):
        lw = tok.lower
        if tok.text.startswith("__"):
            tag = "NUM" if "TIME" in tok.text or "NUM" in tok.text else "X"
        elif not tok.is_word:
            tag = "NUM" if lw[:1].isdigit() or lw[:1] == "$" else "PUNCT"
        elif lw in MODALS:
            tag = "AUX"
        elif lw in AUXILIARIES:
            tag = "AUX"
        elif lw in NEGATIONS and lw not in ("no", "nobody", "none"):
            tag = "PART"
        elif lw in DETERMINERS:
            tag = "DET"
        elif lw in PREPOSITIONS:
            tag = "ADP"
        elif lw in CONJUNCTIONS:
            tag = "SCONJ" if lw in ("unless", "if", "when", "while", "provided", "whenever") else "CCONJ"
        elif lw in PRONOUNS or lw in ("nobody", "none", "everyone", "anyone", "someone"):
            tag = "PRON"
        elif lw in ADVERBS or (lw.endswith("ly") and len(lw) > 4):
            tag = "ADV"
        elif prev in ("AUX", "PART") and (tok.stem in known_verbs or lw.endswith(("ed", "en"))):
            tag = "VERB"
        elif tok.stem in known_verbs and prev not in ("DET", "ADJ"):
            tag = "VERB"
        elif tok.stem in known_nouns:
            tag = "NOUN"
        elif prev in ("AUX", "PART") and i + 1 < len(tokens) and tokens[i + 1].lower in DETERMINERS:
            tag = "VERB"
        elif lw.endswith(VERB_SUFFIXES):
            tag = "VERB"
        elif lw.endswith(ADJ_SUFFIXES) and i + 1 < len(tokens) and tokens[i + 1].is_word:
            tag = "ADJ"
        elif lw.endswith(NOUN_SUFFIXES) or lw.endswith("s"):
            tag = "NOUN"
        else:
            tag = "NOUN"
        tok.pos = tag
        tok.morphemes = segment(tok.text) if tok.is_word else [tok.text]
        prev = tag
    return tokens


def split_sentences(text: str) -> list[tuple[str, int]]:
    """Split policy text into candidate rule sentences, returning (sentence, offset).

    Bullet points and numbered list items each become their own sentence."""
    out: list[tuple[str, int]] = []
    # A period between two digits ("$1.5k") does not end a sentence.
    pattern = re.compile(r"(?:[^.;!?\n]|(?<=\d)\.(?=\d))+(?:[.;!?]+|$)", re.MULTILINE)
    for m in pattern.finditer(text):
        raw = m.group(0)
        stripped = raw.strip()
        lead = len(raw) - len(raw.lstrip())
        bullet = re.match(r"^(?:[-*•]|\d+[.)]|[a-z][.)])\s+", stripped)
        if bullet:
            lead += bullet.end()
            stripped = stripped[bullet.end():]
        if len(stripped.strip(" .;!?")) < 3:
            continue
        out.append((stripped, m.start() + lead))
    return out


def detect_language(text: str) -> tuple[str, float]:
    """Stopword-profile language identification (FIG. 2, step 202)."""
    words = re.findall(r"[a-zA-ZÀ-ÿ]+", text.lower())
    if not words:
        return "en", 0.0
    scores = {lang: sum(1 for w in words if w in sw) for lang, sw in LANGUAGE_STOPWORDS.items()}
    best = max(scores, key=lambda k: scores[k])
    total = sum(scores.values()) or 1
    return best, round(scores[best] / total, 2)
