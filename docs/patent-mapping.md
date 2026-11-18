# Patent-to-code map: US 10,606,957 B1

Raivana implements the system and method of US 10,606,957 B1 element by element.
This document maps each claim and figure to the code that implements it.

## System elements (FIG. 1)

| Patent element | Raivana component | Code |
|---|---|---|
| Natural language policy text (102), from documents, email, web forms or direct entry | CLI argument, file, stdin, REST API, web UI, MCP tool | `cli.py`, `server.py`, `mcp_server.py` |
| **Policy composer (104)** | `PolicyComposer` | `composer.py` |
| **Attribute dictionary (106)** | `AttributeDictionary`, YAML packs (core, healthcare, finance, cloud) plus a user dictionary | `dictionary.py`, `data/dictionary/*.yaml` |
| **Natural language policy sets (108)**, organised by language and by organisation, branch or department | `PolicySets`, sample rules with semantic signatures | `policysets.py`, `data/policysets/*.yaml` |
| Logical access control policy code (110) | XACML 3.0, NGAC, Cedar, Rego generators | `generators/` |
| Attribute data (112) | Request attributes for the policy tester | `ir.AccessRequest`, `analysis.evaluate` |
| **Policies archive (114)** | Versioned file-based archive with hashes and provenance | `archive.py` |

## Claims

| Claim | Requirement | Implementation |
|---|---|---|
| 1, 8, 15 | Receive free-text policy strings | `PolicyComposer.translate()` |
| 1, 8, 15 | Use NLP to identify candidate attributes **and store them in the attribute dictionary** | `RulesEngine._learn_unknowns()` discovers them; `PolicyComposer._commit_learned()` stores them; `raivana translate --save-learned` persists them |
| 1, 8, 15 | Identify candidate rules from the policy set | `RulesEngine.analyze()` builds a semantic signature; `PolicySets.match()` ranks the sample rules |
| 1, 8, 15 | Populate the candidate rules with the candidate attributes | `policysets.populate()` |
| 1, 8, 15 | Determine whether each populated rule is well-formed | `validate.check_rule()` checks types, operators, satisfiability and categories |
| 1, 8, 15 | Translate into logical access control policy code | `generators.generate_all()` |
| 8 | Store the code in a policy archive | `PolicyArchive.save()`, `raivana translate --archive` |
| 2, 9, 16 | XACML output | `generators/xacml.py`, with structural validation and optional XSD validation |
| 3, 10, 17 | NGAC output | `generators/ngac.py` produces a PML script and a JSON graph |
| 4, 11, 18 | Populate the attribute dictionary from the policy set | `AttributeDictionary.learn_from_policy_sets()`, called by `PolicyComposer.__init__` |
| 5, 12, 19 | Populate the attribute dictionary from user input | `raivana dict add`, `AttributeDictionary.add_value()` |
| 6, 13, 20 | Named entity recognition | `RulesEngine._ner()`: longest-match recognition against the dictionary, with salience scores |
| 7, 14 | Morphological segmentation | `nlp.segment()` and `nlp.negating_prefix()`, which recognise forms such as "non-employees" and "unmanaged" |

## Figures

### FIG. 2: Rule identification
| Step | Implementation |
|---|---|
| 202 Identify language | `nlp.detect_language()` |
| 204 Identify policy type | `AttributeDictionary.detect_policy_type()` scores vocabulary overlap per domain pack |
| 206 Select policy sets | `PolicySets.select(language, policy_type, organization)` |
| 208 Identify candidate attributes | NER and learning in `RulesEngine.analyze()` |
| 210 Identify candidate rules | Sentence and clause splitting (`nlp.split_sentences`, `rules.split_clauses`), then signature matching |

### FIG. 3: Rule translation
| Step | Implementation |
|---|---|
| 304 Match the rule to sample rules | `PolicySets.match()`: sequence similarity of signatures, with penalties on key markers |
| 306 Populate the sample | `policysets.populate()`, which expands negated groups with De Morgan's law |
| 308 Well-formed? | `validate.is_well_formed()` |
| 310 Generate code | `generators/` |
| 312 User confirms; a rejection returns to 304 | `PolicyComposer.reject()` tries the next-best sample. Available as `raivana translate -i`, the web UI "reject" button, and `POST /api/sessions/{id}/reject` |
| 314 Export or deploy | Files in `-o`, the archive, and the REST API |

### FIG. 4: Text characteristics
Logic operators (only, unless, except, and/or), negation and combining-algorithm phrases (`rules.detect_combining`). Combining phrases include "denied by default", "first matching rule" and "permits take precedence".

### FIG. 5: XACML translation
NLP covers NER, POS tagging, lemmatisation and morphological segmentation (502). Sample rules are matched by signature (504/506). Conditions are compared against known conditions and the best candidate above the confidence threshold is selected (508). Rules are organised into PolicySet and Policy nodes (510) and exported as XACML 3.0 XML (512).

### FIG. 6: NGAC translation
Subject and resource conditions become user and object attributes. Permits become associations and denies become prohibitions (608). Statements that lack the parameters for a complete NGAC statement are split out and reported (610).

### FIG. 7: User experience
| Step | Implementation |
|---|---|
| 702 Select output format | `--format`, web UI tabs |
| 704 Select input source | text, `--file`, stdin, web form, MCP |
| 706 Define attributes | `raivana dict add`, `--dictionary` |
| 708 Translate | `raivana translate`, Translate button |
| 710/712 Review translation, summary and components | Plain-English summaries (`explain.py`), entity highlighting and rule cards |
| 718 Alter or remove components and regenerate | `PolicyComposer.remove_rule()`, the web UI remove button, interactive review |
| 714/716 Export or deploy | `-o`, `--archive`, downloads |

## Beyond the patent

These features build on the patented method. They are not described in the patent itself:

- **Claude engine**: structured-output IR generation, hybrid routing that sends only hard sentences to the model, multi-model routing with escalation, and prompt caching.
- **Cedar and Rego** generators alongside XACML and NGAC.
- **Policy decision point** with five combining algorithms and full decision traces.
- **Static analysis**: conflicts, overlaps, intentional exceptions, redundancy, unreachable rules and unsatisfiable rules, each with a concrete example request built by a constraint solver.
- **Differential verification**: randomised requests checked for agreement between the reference PDP and the real Cedar engine and OPA.
- **MCP server**: AI agents can author and test access policy.
