"""REST API + web UI (FastAPI).

    raivana serve            # http://127.0.0.1:8000  (interactive API docs at /docs)

Security notes for deployment beyond localhost:
  * set RAIVANA_SERVER_TOKEN to require `Authorization: Bearer <token>` on API calls;
  * set RAIVANA_DISABLE_LLM=1 to force the offline engine (no API spend);
  * request text is capped at RAIVANA_MAX_CHARS (default 20,000).
"""

from __future__ import annotations

import json
import os
import secrets
import uuid
from collections import OrderedDict
from importlib import resources
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import __version__
from .analysis import analyze, evaluate
from .archive import PolicyArchive
from .composer import ENGINES, PolicyComposer, TranslationResult
from .config import load_env
from .engines.llm import MODEL_ALIASES, llm_available
from .generators import FORMATS, generate_all
from .ir import AccessRequest, Policy
from .validate import check_policy

load_env()
MAX_CHARS = int(os.environ.get("RAIVANA_MAX_CHARS", "20000"))
TOKEN = os.environ.get("RAIVANA_SERVER_TOKEN")
LLM_DISABLED = os.environ.get("RAIVANA_DISABLE_LLM") == "1"

app = FastAPI(title="Raivana", version=__version__,
              description="Translate natural-language access control policy into XACML, NGAC, Cedar and Rego. "
                          "Reference implementation of US Patent 10,606,957 B1.")


def auth(authorization: Optional[str] = Header(default=None)) -> None:
    if TOKEN and not (authorization and secrets.compare_digest(authorization, f"Bearer {TOKEN}")):
        raise HTTPException(401, "Missing or invalid bearer token")


# --------------------------------------------------------------------------- sessions (review loop)
_SESSIONS: OrderedDict[str, tuple[PolicyComposer, TranslationResult, list[str]]] = OrderedDict()
_MAX_SESSIONS = 64


def _store(composer: PolicyComposer, result: TranslationResult, formats: list[str]) -> str:
    sid = uuid.uuid4().hex[:12]
    _SESSIONS[sid] = (composer, result, formats)
    while len(_SESSIONS) > _MAX_SESSIONS:
        _SESSIONS.popitem(last=False)
    return sid


def _session(sid: str) -> tuple[PolicyComposer, TranslationResult, list[str]]:
    if sid not in _SESSIONS:
        raise HTTPException(404, "Unknown or expired session; translate again.")
    _SESSIONS.move_to_end(sid)
    return _SESSIONS[sid]


def _payload(sid: str, result: TranslationResult) -> dict[str, Any]:
    return {"session": sid, **result.to_dict(detail=True)}


# --------------------------------------------------------------------------- models
class TranslateBody(BaseModel):
    text: str
    engine: str = "auto"
    model: Optional[str] = None
    route: str = "fixed"
    effort: str = "medium"
    formats: list[str] = Field(default_factory=lambda: list(FORMATS))
    policy_id: Optional[str] = None
    threshold: float = 0.6


class PolicyBody(BaseModel):
    policy: Policy
    formats: list[str] = Field(default_factory=lambda: list(FORMATS))


class EvaluateBody(BaseModel):
    policy: Policy
    request: AccessRequest


class VerifyBody(BaseModel):
    policy: Policy
    n: int = 300


class SentenceBody(BaseModel):
    sentence: int


class RuleBody(BaseModel):
    rule_id: str


# --------------------------------------------------------------------------- routes
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    return (resources.files("raivana") / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"version": __version__, "llm_available": llm_available() and not LLM_DISABLED,
            "engines": list(ENGINES), "formats": FORMATS, "models": MODEL_ALIASES,
            "auth_required": bool(TOKEN)}


@app.get("/api/examples")
def examples() -> list[dict[str, Any]]:
    base = resources.files("raivana") / "data" / "examples"
    out = []
    for name in ("hospital", "bank", "cloud"):
        out.append({"name": name, "text": (base / f"{name}.txt").read_text(encoding="utf-8"),
                    "probes": json.loads((base / f"{name}.probes.json").read_text(encoding="utf-8"))})
    return out


@app.post("/api/translate", dependencies=[Depends(auth)])
def translate(body: TranslateBody) -> dict[str, Any]:
    if len(body.text) > MAX_CHARS:
        raise HTTPException(413, f"Policy text exceeds {MAX_CHARS} characters")
    if body.engine not in ENGINES:
        raise HTTPException(422, f"engine must be one of {ENGINES}")
    engine = "rules" if LLM_DISABLED else body.engine
    formats = [f for f in body.formats if f in FORMATS] or ["json"]
    composer = PolicyComposer(engine=engine, model=body.model, route=body.route, effort=body.effort,
                              threshold=body.threshold)
    result = composer.translate(body.text, policy_id=body.policy_id, formats=formats)
    return _payload(_store(composer, result, formats), result)


@app.post("/api/sessions/{sid}/reject", dependencies=[Depends(auth)])
def reject(sid: str, body: SentenceBody) -> dict[str, Any]:
    composer, result, formats = _session(sid)
    if not 0 <= body.sentence < len(result.sentences):
        raise HTTPException(422, "sentence index out of range")
    result = composer.reject(result, body.sentence, formats)
    _SESSIONS[sid] = (composer, result, formats)
    return _payload(sid, result)


@app.post("/api/sessions/{sid}/remove", dependencies=[Depends(auth)])
def remove(sid: str, body: RuleBody) -> dict[str, Any]:
    composer, result, formats = _session(sid)
    result = composer.remove_rule(result, body.rule_id, formats)
    _SESSIONS[sid] = (composer, result, formats)
    return _payload(sid, result)


@app.post("/api/generate", dependencies=[Depends(auth)])
def generate_route(body: PolicyBody) -> dict[str, Any]:
    comp = PolicyComposer(engine="rules")
    arts = generate_all(body.policy, [f for f in body.formats if f in FORMATS], comp.dictionary)
    return {k: a.to_dict() for k, a in arts.items()}


@app.post("/api/evaluate", dependencies=[Depends(auth)])
def evaluate_route(body: EvaluateBody) -> dict[str, Any]:
    return evaluate(body.policy, body.request).to_dict()


@app.post("/api/analyze", dependencies=[Depends(auth)])
def analyze_route(body: PolicyBody) -> dict[str, Any]:
    comp = PolicyComposer(engine="rules")
    return {"issues": [i.to_dict() for i in check_policy(body.policy, comp.dictionary)],
            "findings": [f.to_dict() for f in analyze(body.policy)]}


@app.post("/api/verify", dependencies=[Depends(auth)])
def verify_route(body: VerifyBody) -> list[dict[str, Any]]:
    from .verify import verify

    return [r.to_dict() for r in verify(body.policy, min(body.n, 2000))]


@app.get("/api/dictionary")
def dictionary() -> dict[str, Any]:
    return PolicyComposer(engine="rules").dictionary.to_json()


@app.get("/api/archive", dependencies=[Depends(auth)])
def archive_list() -> list[dict[str, Any]]:
    return PolicyArchive().list()


@app.post("/api/archive", dependencies=[Depends(auth)])
def archive_save(body: PolicyBody) -> dict[str, Any]:
    comp = PolicyComposer(engine="rules")
    arts = generate_all(body.policy, [f for f in body.formats if f in FORMATS], comp.dictionary)
    return PolicyArchive().save(body.policy, arts, {"via": "api"})
