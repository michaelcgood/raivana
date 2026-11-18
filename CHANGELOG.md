# Changelog

## 0.1.0 - 2026-11-18

First public release.

- Policy composer pipeline implementing US 10,606,957 B1: language and policy-type detection, attribute
  dictionary with learning, natural-language policy sets, signature matching, rule population,
  well-formedness checks, human review loop and policy archive.
- Engines: offline rules engine, Claude engine (structured outputs, multi-model routing with escalation,
  prompt caching, server-side refusal fallback) and hybrid mode.
- Generators: XACML 3.0, NGAC (PML and graph), Cedar, Rego, JSON IR.
- Reference PDP with five combining algorithms and decision traces. Static analysis for conflicts,
  overlaps, redundancy, unreachable and unsatisfiable rules, with example requests.
- Differential verification against the Cedar engine and OPA.
- Interfaces: CLI, Python API, REST API, web UI, MCP server.
- Domain packs: core, healthcare, finance, cloud. 35-case behavioural evaluation suite.
