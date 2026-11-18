# Contributing to Raivana

Thanks for your interest! Raivana is the reference implementation of US 10,606,957 B1, and
contributions are welcome under the Apache License 2.0.

## Development setup

```bash
git clone https://github.com/michaelcgood/raivana && cd raivana
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                      # offline; no API key needed
raivana eval --engine rules # behavioural evaluation suite
ruff check src tests
```

Optional: install [OPA](https://www.openpolicyagent.org/docs/latest/#running-opa) so the Rego
differential tests run locally (CI always runs them).

## Good first contributions

- **Vocabulary packs**: add a domain dictionary in `src/raivana/data/dictionary/` (e.g. education, government, retail).
- **Sample rules**: add phrasing patterns to `src/raivana/data/policysets/`, each with an eval case in `data/evals/cases.yaml`.
- **Eval cases**: real-world policy sentences that Raivana gets wrong are the most valuable bug reports.
- **New targets**: a generator only needs `generate(policy) -> str` and must agree with the reference PDP. Add it to `verify.py`.

## Contributor License Agreement

Before your first pull request can be merged, you must agree to the [Contributor License Agreement](CLA.md)
by ticking the box in the pull request template. The CLA lets the maintainers keep offering Raivana under
Apache-2.0 and in commercial editions. You keep the copyright to your contribution.

## Rules of the road

- Every behaviour change needs a test. Every new phrasing needs an eval case.
- Generators must keep the IR semantics in `docs/architecture.md`. `raivana verify` must stay green.
- Keep the core dependency-light. Optional features go behind extras.
