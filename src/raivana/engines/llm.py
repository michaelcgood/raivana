"""Claude-powered translation engine with multi-model routing.

The language model never writes policy code. It returns a small, schema-validated
intermediate representation (via structured outputs), which Raivana then checks for
well-formedness and compiles deterministically into XACML / NGAC / Cedar / Rego.
That keeps output tokens (the expensive part) to a minimum and makes every generated
policy syntactically valid by construction.

Cost controls:
  * the system prompt (instructions + attribute dictionary + policy-set examples) is
    byte-stable and marked for prompt caching, so repeat calls pay ~5% for it;
  * in hybrid mode only sentences the offline engine could not handle are sent;
  * all unresolved sentences of a document go in one request.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel

from ..dictionary import AttributeDictionary
from ..ir import AttributeRef, Condition, Effect, Operator, Rule
from ..policysets import PolicySets

# --------------------------------------------------------------------------- models & routing
MODEL_ALIASES = {
    "haiku": "claude-haiku-5-5",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
    "fable": "claude-fable-5-2",
}
DEFAULT_MODEL = "claude-sonnet-5-5"
STRONG_MODEL = "claude-opus-5-5"
FAST_MODEL = "claude-haiku-5-5"

# USD per million tokens: (input, output, cache read, cache write)
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10, 2.50),
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-fable-5-2": (8.00, 40.00, 0.20, 10.00),
}
# Models that support the server-side refusal fallback on the first-party API.
FALLBACK_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-2"}
TRICKY = ("unless", "except", "only", "provided", "otherwise", "notwithstanding", "subject to",
          "however", "but", "whereas", "in addition", "regardless")


def resolve_model(name: str | None) -> str:
    if not name:
        return os.environ.get("RAIVANA_MODEL", DEFAULT_MODEL)
    return MODEL_ALIASES.get(name.lower(), name)


@dataclass
class ModelRouter:
    """Chooses the model for a request.

    mode="fixed": always `model` (default Claude Sonnet 5.5), escalating to `strong` once if
                  the output fails validation.
    mode="auto":  route by complexity -- `fast` for short, simple batches, `model` for normal
                  ones, `strong` for long or exception-heavy policy text.
    """

    model: str = DEFAULT_MODEL
    mode: Literal["fixed", "auto"] = "fixed"
    fast: str = FAST_MODEL
    strong: str = STRONG_MODEL
    escalate: bool = True

    def complexity(self, sentences: list[str]) -> float:
        text = " ".join(sentences).lower()
        return len(sentences) * 0.5 + sum(text.count(t) for t in TRICKY) + len(text) / 400

    def choose(self, sentences: list[str]) -> str:
        if self.mode != "auto":
            return self.model
        score = self.complexity(sentences)
        if score < 2.0:
            return self.fast
        if score > 8.0:
            return self.strong
        return self.model


# --------------------------------------------------------------------------- structured output schema
class LLMCondition(BaseModel):
    category: Literal["subject", "resource", "environment"]
    attribute: str
    datatype: Literal["string", "integer", "double", "boolean"]
    operator: Literal["eq", "neq", "in", "not_in", "gt", "gte", "lt", "lte"]
    values: list[str]
    ref_category: Optional[Literal["subject", "resource", "environment"]]
    ref_attribute: Optional[str]


class LLMRule(BaseModel):
    effect: Literal["Permit", "Deny"]
    actions: list[str]
    conditions: list[LLMCondition]
    explanation: str


class LLMSentence(BaseModel):
    index: int
    is_rule: bool
    confidence: float
    rules: list[LLMRule]


class LLMAttribute(BaseModel):
    category: Literal["subject", "resource", "environment", "action"]
    attribute: str
    datatype: Literal["string", "integer", "double", "boolean"]
    value: Optional[str]
    synonyms: list[str]


class LLMResult(BaseModel):
    sentences: list[LLMSentence]
    new_attributes: list[LLMAttribute]
    default_effect: Optional[Literal["Permit", "Deny"]]
    combining: Optional[Literal["deny-overrides", "permit-overrides", "first-applicable"]]
    ambiguities: list[str]


SYSTEM_INSTRUCTIONS = """You are Raivana's policy translator. You convert natural-language access control \
policy statements into a precise attribute-based intermediate representation (IR). Raivana compiles \
your IR into XACML 3.0, NGAC, Cedar and Rego, so precision matters more than prose.

IR SEMANTICS
- A rule has an effect (Permit or Deny), a list of actions (OR-ed; [] means ANY action) and a list \
of conditions (AND-ed).
- Condition = category.attribute OPERATOR values. Operators: eq, neq, in, not_in, gt, gte, lt, lte.
  Use "in"/"not_in" for several alternative values of ONE attribute ("doctors and nurses" -> role in \
[doctor, nurse]). Ordering operators need integer/double attributes.
- Attribute-to-attribute comparisons: set ref_category/ref_attribute and leave values empty \
(e.g. "their own records" -> resource.owner eq ref subject.id; "in their department" -> \
resource.department eq ref subject.department). Otherwise set both ref fields to null.
- Values are strings in the JSON; write booleans as "true"/"false" and numbers as digits \
("$10,000" -> "10000").
- Times of day use environment.hour (integer 0-23): "9am-5pm" -> hour gte 9 AND hour lt 17; \
"business hours" means 9-17; overnight windows use not_in with the excluded hours. Days use \
environment.day_of_week with values mon..sun.
- Unless/except/only phrasing usually needs TWO rules: e.g. "Only doctors can prescribe medication" -> \
Permit role eq doctor; Deny role neq doctor (same action/resource). "X can do Y unless Z" -> Permit \
X,Y; Deny X,Y,Z. "X cannot do Y unless Z" -> Deny X,Y with NOT Z; Permit X,Y,Z. Negating several \
conditions requires one Deny rule per negated condition (De Morgan).
- Rules are combined with deny-overrides unless the text says otherwise. If the text states a default \
("everything else is denied"), set default_effect; that sentence itself is not a rule.

VOCABULARY
Prefer the canonical attributes, values and actions in the dictionary below. Use snake_case for any \
new value. When you must introduce an attribute, value or action that is not in the dictionary, \
also list it in new_attributes (category "action" for new actions) with its surface synonyms.

OUTPUT
- Return one entry in `sentences` for EVERY input sentence index, in order.
- Non-rules (headings, definitions, rationale) get is_rule=false and no rules.
- confidence: 0..1, your confidence that the rules capture the sentence exactly.
- `explanation`: one short sentence restating the rule in plain English.
- List genuine ambiguities (e.g. undefined terms) in `ambiguities`; do not invent facts.
"""


@dataclass
class LLMUsage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    requests: int = 0
    escalated: bool = False
    fallback: bool = False

    @property
    def cost_usd(self) -> float:
        pi, po, pr, pw = PRICES.get(self.model, PRICES[DEFAULT_MODEL])
        return round((self.input_tokens * pi + self.output_tokens * po + self.cache_read_tokens * pr
                      + self.cache_write_tokens * pw) / 1_000_000, 6)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["cost_usd"] = self.cost_usd
        return d


@dataclass
class LLMTranslation:
    rules: dict[int, list[Rule]] = field(default_factory=dict)
    non_rules: set[int] = field(default_factory=set)
    confidence: dict[int, float] = field(default_factory=dict)
    new_attributes: list[dict[str, Any]] = field(default_factory=list)
    default_effect: str | None = None
    combining: str | None = None
    ambiguities: list[str] = field(default_factory=list)
    usage: list[LLMUsage] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def llm_available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.environ.get("RAIVANA_PROVIDER") in ("bedrock", "vertex", "foundry"))


def make_client(provider: str | None = None) -> Any:
    """Create an Anthropic SDK client for the chosen endpoint."""
    import anthropic

    provider = (provider or os.environ.get("RAIVANA_PROVIDER") or "anthropic").lower()
    if provider == "bedrock":
        return anthropic.AnthropicBedrockMantle(aws_region=os.environ.get("AWS_REGION", "us-east-1"))
    if provider == "vertex":
        return anthropic.AnthropicVertex(project_id=os.environ["GOOGLE_CLOUD_PROJECT"],
                                         region=os.environ.get("GOOGLE_CLOUD_REGION", "global"))
    if provider == "foundry":
        return anthropic.AnthropicFoundry(api_key=os.environ.get("ANTHROPIC_FOUNDRY_API_KEY"),
                                          resource=os.environ["ANTHROPIC_FOUNDRY_RESOURCE"])
    return anthropic.Anthropic()


class ClaudeEngine:
    name = "claude"

    def __init__(self, dictionary: AttributeDictionary, policy_sets: PolicySets,
                 router: ModelRouter | None = None, effort: str = "medium", provider: str | None = None,
                 examples: list[tuple[str, str]] | None = None, client: Any = None) -> None:
        self.dictionary = dictionary
        self.policy_sets = policy_sets
        self.router = router or ModelRouter()
        self.effort = effort
        self.provider = (provider or os.environ.get("RAIVANA_PROVIDER") or "anthropic").lower()
        self.examples = examples or []
        self._client = client
        self._system: str | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_client(self.provider)
        return self._client

    def system_prompt(self) -> str:
        """Built once per engine and then frozen, so every request shares a byte-identical,
        cacheable prefix (vocabulary learned mid-session does not invalidate the cache)."""
        if self._system is None:
            self._system = self._build_system_prompt()
        return self._system

    def _build_system_prompt(self) -> str:
        parts = [SYSTEM_INSTRUCTIONS, "ATTRIBUTE DICTIONARY", self.dictionary.to_prompt()]
        if self.examples:
            parts.append("EXAMPLES (policy-set sample rules and their correct IR)")
            for text, ir in self.examples:
                parts.append(f"Input: {text}\nIR: {ir}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ call
    def _call(self, model: str, user: str) -> tuple[LLMResult | None, LLMUsage, str | None]:
        usage = LLMUsage(model=model)
        system = [{"type": "text", "text": self.system_prompt(), "cache_control": {"type": "ephemeral"}}]
        kwargs: dict[str, Any] = dict(
            model=model, max_tokens=16000, system=system,
            messages=[{"role": "user", "content": user}],
            output_format=LLMResult,
            output_config={"effort": self.effort},
        )
        import anthropic

        try:
            if self.provider == "anthropic" and model in FALLBACK_MODELS:
                resp = self.client.beta.messages.parse(
                    **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            else:
                resp = self.client.messages.parse(**kwargs)
        except anthropic.AuthenticationError:
            return None, usage, "Authentication failed: check ANTHROPIC_API_KEY."
        except anthropic.RateLimitError:
            return None, usage, "Rate limited by the Claude API; try again shortly."
        except anthropic.BadRequestError as e:
            return None, usage, f"Claude API rejected the request: {e.message}"
        except anthropic.APIStatusError as e:
            return None, usage, f"Claude API error {e.status_code}: {e.message}"
        except anthropic.APIConnectionError:
            return None, usage, "Could not reach the Claude API (network error)."
        u = resp.usage
        usage.requests = 1
        usage.input_tokens = u.input_tokens or 0
        usage.output_tokens = u.output_tokens or 0
        usage.cache_read_tokens = getattr(u, "cache_read_input_tokens", 0) or 0
        usage.cache_write_tokens = getattr(u, "cache_creation_input_tokens", 0) or 0
        usage.model = getattr(resp, "model", model) or model
        usage.fallback = usage.model != model
        if resp.stop_reason == "refusal":
            return None, usage, "The model declined to translate this text."
        if resp.stop_reason == "max_tokens":
            return None, usage, "Output was truncated (max_tokens); split the policy into smaller parts."
        parsed = getattr(resp, "parsed_output", None)
        if parsed is None:
            return None, usage, "The model returned no structured output."
        return parsed, usage, None

    def translate(self, sentences: list[tuple[int, str]], context: str = "") -> LLMTranslation:
        """Translate (index, sentence) pairs. Indices are preserved in the result."""
        out = LLMTranslation()
        if not sentences:
            return out
        listing = "\n".join(f"[{i}] {s}" for i, s in sentences)
        user = (f"Policy context: {context}\n\n" if context else "") + \
            f"Translate each numbered sentence into IR rules.\n\n{listing}"
        model = self.router.choose([s for _, s in sentences])
        result, usage, err = self._call(model, user)
        out.usage.append(usage)
        problems = err
        if result is not None:
            problems = self._ingest(result, {i for i, _ in sentences}, out)
        if problems and self.router.escalate and model != self.router.strong:
            retry, usage2, err2 = self._call(self.router.strong, user)
            usage2.escalated = True
            out.usage.append(usage2)
            if retry is not None:
                fresh = LLMTranslation(usage=out.usage)
                if not self._ingest(retry, {i for i, _ in sentences}, fresh):
                    return fresh
            if err2:
                out.errors.append(err2)
        if problems:
            out.errors.append(problems)
        return out

    # ------------------------------------------------------------------ conversion
    def _ingest(self, result: LLMResult, expected: set[int], out: LLMTranslation) -> str | None:
        problems: list[str] = []
        for s in result.sentences:
            if s.index not in expected:
                continue
            out.confidence[s.index] = max(0.0, min(1.0, s.confidence))
            if not s.is_rule:
                out.non_rules.add(s.index)
                continue
            rules = []
            for lr in s.rules:
                try:
                    rules.append(self._to_rule(lr))
                except (ValueError, KeyError) as e:
                    problems.append(f"sentence {s.index}: {e}")
            out.rules[s.index] = rules
        missing = expected - set(out.rules) - out.non_rules
        if missing:
            problems.append(f"no translation returned for sentence(s) {sorted(missing)}")
        out.new_attributes = [a.model_dump() for a in result.new_attributes]
        out.default_effect = result.default_effect
        out.combining = result.combining
        out.ambiguities = list(result.ambiguities)
        return "; ".join(problems) if problems else None

    def _to_rule(self, lr: LLMRule) -> Rule:
        rule = Rule(id="", effect=Effect(lr.effect), actions=[a for a in lr.actions if a not in ("*", "any")],
                    description=lr.explanation, engine="claude")
        for c in lr.conditions:
            known = self.dictionary.get(c.attribute)
            dt = c.datatype
            if known and known.datatype != dt and c.ref_attribute is None:
                dt = known.datatype
            ref = AttributeRef(category=c.category, id=c.attribute, datatype=dt)
            op = Operator(c.operator)
            if c.ref_attribute and c.ref_category:
                value: Any = AttributeRef(category=c.ref_category, id=c.ref_attribute, datatype=dt)
            else:
                vals = [self._coerce(v, dt) for v in c.values]
                if not vals:
                    raise ValueError(f"condition on {ref.path} has no value")
                value = vals if op in (Operator.IN, Operator.NOT_IN) else vals[0]
                if op not in (Operator.IN, Operator.NOT_IN) and len(vals) > 1:
                    op = Operator.IN if op == Operator.EQ else Operator.NOT_IN if op == Operator.NEQ else op
                    value = vals if op in (Operator.IN, Operator.NOT_IN) else vals[0]
            cond = Condition(attribute=ref, operator=op, value=value)
            getattr(rule, c.category).append(cond)
        return rule

    @staticmethod
    def _coerce(v: str, dt: str) -> Any:
        s = str(v).strip()
        if dt == "boolean":
            return s.lower() in ("true", "yes", "1")
        if dt == "integer":
            return int(float(s.replace(",", "").replace("$", "")))
        if dt == "double":
            return float(s.replace(",", "").replace("$", ""))
        return s
