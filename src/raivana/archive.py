"""Policies archive (US 10,606,957 B1, element 114).

A simple, versioned, file-based store. Each policy gets a directory; each export
creates an immutable version folder containing the IR, the source text, the
generated artifacts, and a manifest with hashes and provenance.

    <root>/<policy-id>/v0001/policy.raivana.json
                            /source.txt
                            /<policy-id>.xml, .cedar, .rego, .pml
                            /manifest.json
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .generators import Artifact
from .ir import Policy


def default_root() -> Path:
    return Path(os.environ.get("RAIVANA_ARCHIVE", Path.cwd() / ".raivana" / "archive"))


class PolicyArchive:
    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root) if root else default_root()

    def _dir(self, policy_id: str) -> Path:
        return self.root / policy_id

    def versions(self, policy_id: str) -> list[str]:
        d = self._dir(policy_id)
        return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []

    def save(self, policy: Policy, artifacts: dict[str, Artifact] | None = None,
             provenance: dict[str, Any] | None = None) -> dict[str, Any]:
        versions = self.versions(policy.id)
        latest = self.load_manifest(policy.id) if versions else None
        if latest and latest.get("fingerprint") == policy.fingerprint():
            return latest | {"unchanged": True}
        version = f"v{len(versions) + 1:04d}"
        vdir = self._dir(policy.id) / version
        vdir.mkdir(parents=True, exist_ok=False)
        files: dict[str, str] = {}

        def put(name: str, content: str) -> None:
            (vdir / name).write_text(content, encoding="utf-8")
            files[name] = hashlib.sha256(content.encode()).hexdigest()

        put("policy.raivana.json", policy.model_dump_json(indent=2))
        if policy.source_text:
            put("source.txt", policy.source_text)
        for art in (artifacts or {}).values():
            if art.format != "json":
                put(art.filename, art.content)
        manifest = {
            "policy_id": policy.id, "version": version, "fingerprint": policy.fingerprint(),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rules": len(policy.rules), "files": files, "provenance": provenance or {},
        }
        (vdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def list(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        out = []
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and self.versions(d.name):
                m = self.load_manifest(d.name)
                out.append({"policy_id": d.name, "versions": len(self.versions(d.name)),
                            "latest": m.get("version"), "created": m.get("created"), "rules": m.get("rules")})
        return out

    def load_manifest(self, policy_id: str, version: str | None = None) -> dict[str, Any]:
        version = version or self.versions(policy_id)[-1]
        return json.loads((self._dir(policy_id) / version / "manifest.json").read_text(encoding="utf-8"))

    def load(self, policy_id: str, version: str | None = None) -> Policy:
        version = version or self.versions(policy_id)[-1]
        return Policy.load(str(self._dir(policy_id) / version / "policy.raivana.json"))
