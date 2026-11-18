"""Raivana command-line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from . import __version__
from .analysis import analyze, evaluate
from .archive import PolicyArchive
from .composer import ENGINES, PolicyComposer, TranslationResult
from .explain import describe_rule
from .generators import FORMATS, generate
from .ir import Policy
from .validate import check_policy

console = Console()
LEXERS = {"xacml": "xml", "cedar": "rust", "rego": "ruby", "ngac": "javascript", "json": "json"}
EFFECT_STYLE = {"Permit": "bold green", "Deny": "bold red", "NotApplicable": "yellow"}


def _formats(value: str) -> list[str]:
    if value in ("all", "*"):
        return FORMATS
    fmts = [f.strip().lower() for f in value.split(",") if f.strip()]
    bad = [f for f in fmts if f not in FORMATS]
    if bad:
        raise click.BadParameter(f"unknown format(s) {bad}; choose from {', '.join(FORMATS)} or 'all'")
    return fmts


def _load_policy(ref: str) -> Policy:
    p = Path(ref)
    if p.exists():
        return Policy.load(str(p))
    arch = PolicyArchive()
    if arch.versions(ref):
        return arch.load(ref)
    raise click.ClickException(f"No policy file or archived policy named '{ref}'.")


def _parse_kv(items: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise click.BadParameter(f"expected key=value, got '{item}'")
        k, v = item.split("=", 1)
        val: Any = v
        if v.lower() in ("true", "false"):
            val = v.lower() == "true"
        else:
            try:
                val = int(v)
            except ValueError:
                try:
                    val = float(v)
                except ValueError:
                    val = [x for x in v.split("|")] if "|" in v else v
        out[k] = val
    return out


# =========================================================================== rendering
def render_result(result: TranslationResult, show_code: list[str], verbose: bool = False) -> None:
    pol = result.policy
    head = Text.assemble(
        ("Policy ", "bold"), (pol.id, "bold cyan"), "   ",
        ("language ", "dim"), (f"{result.language}", "white"), "   ",
        ("type ", "dim"), (result.policy_type, "white"), "   ",
        ("engine ", "dim"), (result.engine, "white"), "   ",
        ("combining ", "dim"), (pol.combining.value + (f" / default {pol.default_effect.value}"
                                                       if pol.default_effect else ""), "white"))
    console.print(Panel(head, title="[bold]Raivana[/] translation", border_style="cyan"))

    t = Table(show_header=True, header_style="bold", expand=True, box=None, padding=(0, 1))
    t.add_column("#", style="dim", width=3)
    t.add_column("Policy text", ratio=3)
    t.add_column("Status", width=11)
    t.add_column("Via", width=22)
    t.add_column("Conf", width=5, justify="right")
    t.add_column("Rules", width=10)
    for s in result.sentences:
        style = {"translated": "green", "unresolved": "red", "meta": "blue"}.get(s.status, "dim")
        via = s.engine or "-"
        if s.template:
            via = f"{via}:{s.template}"
        t.add_row(str(s.index), s.text, f"[{style}]{s.status}[/]", via,
                  f"{s.confidence:.2f}" if s.confidence else "", ", ".join(s.rule_ids))
    console.print(t)

    console.print()
    for r in pol.rules:
        mark = "[green]PERMIT[/]" if r.effect.value == "Permit" else "[red]DENY  [/]"
        console.print(f"  {mark} [bold]{r.id:>4}[/]  {describe_rule(r)}", highlight=False)
    if verbose:
        for s in result.sentences:
            if s.analysis and s.analysis.entities:
                ents = ", ".join(f"{e.text}->{e.attribute.id if e.attribute else e.kind}"
                                 f"{'=' + str(e.value) if e.value else ''}" for e in s.analysis.entities)
                console.print(f"  [dim]#{s.index} signature={s.analysis.signature} entities: {ents}[/]")

    if result.findings:
        console.print()
        for f in result.findings:
            color = {"info": "blue", "warning": "yellow", "error": "red"}[f.severity]
            console.print(f"  [{color}]{f.kind:>11}[/] {f.message}")
    errors = [i for i in result.issues if i.severity == "error"]
    warns = [i for i in result.issues if i.severity == "warning"]
    if errors or warns:
        console.print()
    for i in errors:
        console.print(f"  [red]error[/] {i.rule or ''} {i.message}")
    for i in warns:
        console.print(f"  [yellow]warn [/] {i.rule or ''} {i.message}")
    for a in result.ambiguities:
        console.print(f"  [magenta]ambiguity[/] {a}")
    if result.learned:
        console.print()
        for item in result.learned:
            console.print(f"  [cyan]learned[/] {item.get('attribute') or 'action'} = {item['value']} "
                          f"[dim](from '{item.get('surface', '')}', {item.get('source', 'nlp')})[/]")
    if result.usage:
        tok_in = sum(u.input_tokens + u.cache_read_tokens + u.cache_write_tokens for u in result.usage)
        tok_out = sum(u.output_tokens for u in result.usage)
        models = ", ".join(sorted({u.model for u in result.usage}))
        console.print(f"\n  [dim]Claude: {models} | {tok_in} in / {tok_out} out tokens | "
                      f"est. cost ${result.cost_usd:.4f}[/]")
    for fmt in show_code:
        art = result.artifacts.get(fmt)
        if not art:
            continue
        console.print()
        console.print(Panel(Syntax(art.content, LEXERS.get(fmt, "text"), word_wrap=True, theme="monokai"),
                            title=f"{fmt} - {art.filename}", border_style="dim"))
        for w in art.warnings:
            console.print(f"  [yellow]{fmt} warning[/] {w}")


def write_artifacts(result: TranslationResult, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for art in result.artifacts.values():
        p = outdir / art.filename
        p.write_text(art.content, encoding="utf-8")
        written.append(p)
    return written


# =========================================================================== commands
@click.group(context_settings={"help_option_names": ["-h", "--help"]},
             epilog="Enterprise connectors & implementation: https://vectoraccessgroup.com")
@click.version_option(__version__, prog_name="raivana")
def main() -> None:
    """Raivana - translate natural-language access policy into XACML, NGAC, Cedar and Rego.

    Reference implementation of US Patent 10,606,957 B1.
    """


@main.command()
@click.argument("text", required=False)
@click.option("-f", "--file", "file_", type=click.Path(exists=True, dir_okay=False), help="Read policy text from a file.")
@click.option("-o", "--out", type=click.Path(file_okay=False), help="Write generated files to this directory.")
@click.option("-F", "--format", "fmt", default="json,xacml,ngac,cedar,rego", show_default=True,
              help=f"Comma-separated output formats ({', '.join(FORMATS)}) or 'all'.")
@click.option("-e", "--engine", type=click.Choice(ENGINES), default="auto", show_default=True,
              help="rules = offline; claude = LLM only; hybrid = offline first, Claude for the rest.")
@click.option("-m", "--model", default=None, help="Claude model id or alias (haiku|sonnet|opus|fable). Default: sonnet.")
@click.option("--route", type=click.Choice(["fixed", "auto"]), default="fixed", show_default=True,
              help="auto = pick Haiku/Sonnet/Opus by policy complexity.")
@click.option("--effort", type=click.Choice(["low", "medium", "high", "xhigh", "max"]), default="medium", show_default=True)
@click.option("--provider", type=click.Choice(["anthropic", "bedrock", "vertex", "foundry"]), default=None,
              help="Claude endpoint (default: anthropic or $RAIVANA_PROVIDER).")
@click.option("--policy-id", default=None, help="Policy identifier (default: derived from the text).")
@click.option("--threshold", type=float, default=0.6, show_default=True, help="Minimum confidence for offline matches.")
@click.option("--dictionary", "dictionaries", multiple=True, type=click.Path(exists=True),
              help="Extra attribute dictionary YAML (repeatable).")
@click.option("--policy-set", "policy_sets", multiple=True, type=click.Path(exists=True),
              help="Extra natural-language policy set YAML file or directory (repeatable).")
@click.option("-i", "--interactive", is_flag=True, help="Review each translation (accept / reject / drop).")
@click.option("--archive", "do_archive", is_flag=True, help="Store the result in the policy archive.")
@click.option("--save-learned", is_flag=True, help="Persist newly learned vocabulary to ~/.raivana/dictionary.yaml.")
@click.option("--show", "show", default=None, help="Formats to print (default: cedar unless -o is given).")
@click.option("--json", "as_json", is_flag=True, help="Print the full result as JSON.")
@click.option("-v", "--verbose", is_flag=True, help="Show NLP signatures and entities.")
def translate(text: str | None, file_: str | None, out: str | None, fmt: str, engine: str, model: str | None,
              route: str, effort: str, provider: str | None, policy_id: str | None, threshold: float,
              dictionaries: tuple[str, ...], policy_sets: tuple[str, ...], interactive: bool, do_archive: bool,
              save_learned: bool, show: str | None, as_json: bool, verbose: bool) -> None:
    """Translate natural-language policy TEXT (or --file, or stdin) into access control policy code."""
    if file_:
        text = Path(file_).read_text(encoding="utf-8")
    elif not text and not sys.stdin.isatty():
        text = sys.stdin.read()
    if not text:
        raise click.UsageError("Provide policy TEXT, --file, or pipe text on stdin.")
    formats = _formats(fmt)
    composer = PolicyComposer(engine=engine, model=model, route=route, effort=effort, provider=provider,
                              threshold=threshold, extra_dictionaries=dictionaries, extra_policy_sets=policy_sets)
    if console.is_terminal and not as_json:
        # ASCII spinner: braille spinners crash legacy Windows consoles (cp1252).
        with console.status("Translating...", spinner="line"):
            result = composer.translate(text, policy_id=policy_id, formats=formats)
    else:
        result = composer.translate(text, policy_id=policy_id, formats=formats)

    if interactive:
        result = review_loop(composer, result, formats)

    if as_json:
        click.echo(json.dumps(result.to_dict(detail=verbose), indent=2, default=str))
    else:
        shown = _formats(show) if show else ([] if out else [f for f in ("cedar",) if f in formats])
        render_result(result, shown, verbose)
    if out:
        paths = write_artifacts(result, Path(out))
        if not as_json:
            console.print("\n  [green]wrote[/] " + ", ".join(str(p) for p in paths))
    if do_archive:
        m = PolicyArchive().save(result.policy, result.artifacts,
                                 {"engine": result.engine, "models": [u.model for u in result.usage]})
        if not as_json:
            console.print(f"  [green]archived[/] {m['policy_id']} {m['version']}"
                          + (" (unchanged)" if m.get("unchanged") else ""))
    if save_learned and result.learned:
        p = composer.save_learned()
        if not as_json:
            console.print(f"  [green]saved learned vocabulary[/] to {p}")
    if any(i.severity == "error" for i in result.issues):
        sys.exit(2)


def review_loop(composer: PolicyComposer, result: TranslationResult, formats: list[str]) -> TranslationResult:
    """Human-in-the-loop confirmation (patent FIG. 3 step 312 and FIG. 7 steps 710-718)."""
    console.print(Panel("Review each translated sentence: [bold]a[/]ccept, [bold]r[/]eject (try the next "
                        "matching sample rule), [bold]d[/]rop, accept [bold]A[/]ll remaining.", border_style="cyan"))
    idx = 0
    accept_all = False
    while idx < len(result.sentences):
        s = result.sentences[idx]
        if s.status != "translated" or accept_all:
            idx += 1
            continue
        console.print(f"\n[bold]#{s.index}[/] {s.text}")
        by_id = {r.id: r for r in result.policy.rules}
        for rid in s.rule_ids:
            r = by_id[rid]
            console.print(f"   [{EFFECT_STYLE[r.effect.value]}]{r.effect.value:6}[/] {describe_rule(r)}")
        choice = click.prompt("   accept/reject/drop/All", default="a",
                              type=click.Choice(["a", "r", "d", "A"]), show_choices=False)
        if choice == "A":
            accept_all = True
        elif choice == "r":
            result = composer.reject(result, s.index, formats)
            continue  # re-review the same sentence with its next candidate
        elif choice == "d":
            for rid in list(s.rule_ids):
                result = composer.remove_rule(result, rid, formats)
            s.status = "unresolved"
            s.notes.append("Dropped by reviewer.")
        idx += 1
    return composer.finalize(result, formats)


@main.command()
@click.argument("policy")
@click.option("-r", "--request", "request_json", help="Request as JSON.")
@click.option("-t", "--tests", "tests_file", type=click.Path(exists=True), help="YAML/JSON file of test cases.")
@click.option("-s", "--subject", multiple=True, help="Subject attribute key=value (repeatable).")
@click.option("-a", "--action", default=None, help="Action name.")
@click.option("-R", "--resource", multiple=True, help="Resource attribute key=value (repeatable).")
@click.option("-E", "--env", multiple=True, help="Environment attribute key=value (repeatable).")
@click.option("--json", "as_json", is_flag=True)
def test(policy: str, request_json: str | None, tests_file: str | None, subject: tuple[str, ...],
         action: str | None, resource: tuple[str, ...], env: tuple[str, ...], as_json: bool) -> None:
    """Evaluate access requests against POLICY (a .raivana.json file or archived policy id).

    \b
    raivana test hospital.raivana.json -s role=nurse -a read -R type=medical_record -E hour=10
    raivana test hospital.raivana.json --tests cases.yaml
    """
    pol = _load_policy(policy)
    if tests_file:
        import yaml

        cases = yaml.safe_load(Path(tests_file).read_text(encoding="utf-8"))
        cases = cases.get("tests", cases) if isinstance(cases, dict) else cases
        failures = 0
        t = Table(box=None, header_style="bold")
        for col in ("", "Test", "Expected", "Got", "Rules"):
            t.add_column(col)
        for i, case in enumerate(cases):
            d = evaluate(pol, case["request"])
            ok = d.decision == case["expect"]
            failures += not ok
            t.add_row("[green]PASS[/]" if ok else "[red]FAIL[/]", case.get("name", f"case {i}"), case["expect"],
                      f"[{EFFECT_STYLE[d.decision]}]{d.decision}[/]", ", ".join(d.determining_rules))
        if as_json:
            click.echo(json.dumps({"total": len(cases), "failures": failures}))
        else:
            console.print(t)
            console.print(f"\n{len(cases) - failures}/{len(cases)} passed")
        sys.exit(1 if failures else 0)
    if request_json:
        req = json.loads(request_json)
    else:
        if not action:
            raise click.UsageError("Provide --request JSON, --tests FILE, or at least --action.")
        req = {"subject": _parse_kv(subject), "action": action, "resource": _parse_kv(resource),
               "environment": _parse_kv(env)}
    d = evaluate(pol, req)
    if as_json:
        click.echo(json.dumps(d.to_dict(), indent=2, default=str))
        return
    console.print(Panel(Text(d.decision, style=EFFECT_STYLE[d.decision], justify="center"),
                        title="Decision", subtitle=d.reason, border_style=EFFECT_STYLE[d.decision].split()[-1]))
    t = Table(box=None, header_style="bold")
    for col in ("Rule", "Effect", "Applies", "Conditions"):
        t.add_column(col)
    for tr in d.trace:
        conds = "; ".join(f"{'[green]Y[/]' if c.result else '[red]N[/]'} {c.condition} (got {c.actual!r})"
                          for c in tr.conditions)
        act = "" if tr.action_match else "[red]N[/] action  "
        t.add_row(tr.rule_id, tr.effect, "[green]yes[/]" if tr.applicable else "[dim]no[/]", act + conds)
    console.print(t)


@main.command()
@click.argument("policy")
@click.option("--json", "as_json", is_flag=True)
def check(policy: str, as_json: bool) -> None:
    """Validate POLICY and detect conflicts, redundancies and impossible rules."""
    pol = _load_policy(policy)
    comp = PolicyComposer(engine="rules")
    issues = check_policy(pol, comp.dictionary)
    findings = analyze(pol)
    if as_json:
        click.echo(json.dumps({"issues": [i.to_dict() for i in issues],
                               "findings": [f.to_dict() for f in findings]}, indent=2, default=str))
    else:
        for i in issues:
            console.print(f"[{'red' if i.severity == 'error' else 'yellow'}]{i.severity}[/] {i.rule or ''} {i.message}")
        for f in findings:
            console.print(f"[{'blue' if f.severity == 'info' else 'yellow'}]{f.kind}[/] {f.message}")
            if f.witness and f.kind == "conflict":
                console.print(f"    [dim]example request: {json.dumps(f.witness)}[/]")
        if not issues and not findings:
            console.print("[green]No issues found.[/]")
    sys.exit(2 if any(i.severity == "error" for i in issues) else 0)


@main.command()
@click.argument("policy")
@click.option("-F", "--format", "fmt", default="all", show_default=True)
@click.option("-o", "--out", type=click.Path(file_okay=False))
def generate_cmd(policy: str, fmt: str, out: str | None) -> None:
    """Generate policy code from a POLICY IR file (e.g. after hand-editing it)."""
    pol = _load_policy(policy)
    comp = PolicyComposer(engine="rules")
    for f in _formats(fmt):
        art = generate(pol, f, comp.dictionary)
        if out:
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / art.filename).write_text(art.content, encoding="utf-8")
            console.print(f"[green]wrote[/] {Path(out) / art.filename}")
        else:
            console.print(Syntax(art.content, LEXERS.get(f, "text"), theme="monokai"))


generate_cmd.name = "generate"


@main.command()
@click.argument("policy")
def explain(policy: str) -> None:
    """Print a plain-English summary of POLICY."""
    pol = _load_policy(policy)
    console.print(f"[bold]{pol.id}[/] ({pol.combining.value}"
                  + (f", default {pol.default_effect.value}" if pol.default_effect else "") + ")")
    for r in pol.rules:
        console.print(f"  [{EFFECT_STYLE[r.effect.value]}]{r.id:>4}[/] {describe_rule(r)}")


@main.command()
@click.argument("policy")
@click.option("-n", "--requests", "n", default=300, show_default=True, help="Number of random requests.")
@click.option("--seed", default=7, show_default=True)
def verify(policy: str, n: int, seed: int) -> None:
    """Differentially verify generated Cedar/Rego against the reference PDP on random requests."""
    from .verify import verify as run_verify

    pol = _load_policy(policy)
    bad = False
    for rep in run_verify(pol, n, seed):
        if not rep.available or rep.skipped_reason:
            console.print(f"[dim]{rep.engine}: skipped - {rep.skipped_reason}[/]")
        elif rep.error:
            bad = True
            console.print(f"[red]{rep.engine}: error[/] {rep.error}")
        elif rep.mismatches:
            bad = True
            console.print(f"[red]{rep.engine}: {len(rep.mismatches)} mismatches[/] of {rep.checked}")
            console.print(json.dumps(rep.mismatches[0], indent=2, default=str))
        else:
            console.print(f"[green]{rep.engine}: {rep.checked}/{rep.checked} decisions agree[/]")
    sys.exit(1 if bad else 0)


# --------------------------------------------------------------------------- dictionary
@main.group("dict")
def dict_group() -> None:
    """Inspect and extend the attribute dictionary."""


@dict_group.command("list")
@click.option("-a", "--attribute", default=None, help="Show only this attribute.")
def dict_list(attribute: str | None) -> None:
    comp = PolicyComposer(engine="rules")
    d = comp.dictionary
    t = Table(header_style="bold", box=None)
    for col in ("Attribute", "Category", "Type", "Values (synonyms)"):
        t.add_column(col)
    for a in sorted(d.attributes.values(), key=lambda a: a.id):
        if attribute and a.id != attribute:
            continue
        vals = "; ".join(f"[cyan]{v}[/] ({', '.join(s[:3])})" for v, s in sorted(a.values.items()))
        t.add_row(a.id, "|".join(a.categories), a.datatype, vals or "[dim]free-form[/]")
    console.print(t)
    if not attribute:
        console.print("\n[bold]Actions[/]: " + ", ".join(f"[cyan]{k}[/]" for k in sorted(d.actions)))


@dict_group.command("add")
@click.argument("attribute")
@click.argument("value")
@click.argument("synonyms", required=False, default="")
@click.option("-c", "--category", type=click.Choice(["subject", "resource", "environment", "action"]), default="subject")
@click.option("--path", type=click.Path(dir_okay=False), default=None, help="Dictionary file (default ~/.raivana/dictionary.yaml).")
def dict_add(attribute: str, value: str, synonyms: str, category: str, path: str | None) -> None:
    """Add VALUE (with comma-separated SYNONYMS) to ATTRIBUTE. Use attribute 'action' to add an action."""
    comp = PolicyComposer(engine="rules")
    syns = [s.strip() for s in synonyms.split(",") if s.strip()]
    if attribute == "action" or category == "action":
        comp.dictionary.add_action(value, syns or [value.replace("_", " ")], source="user")
    else:
        comp.dictionary.add_value(attribute, value, syns or [value.replace("_", " ")], category=category,  # type: ignore[arg-type]
                                  source="user")
    p = comp.save_learned(path)
    console.print(f"[green]added[/] {attribute}={value} -> {p}")


# --------------------------------------------------------------------------- archive
@main.group("archive")
def archive_group() -> None:
    """Browse the policy archive."""


@archive_group.command("list")
def archive_list() -> None:
    rows = PolicyArchive().list()
    if not rows:
        console.print("[dim]Archive is empty. Use `raivana translate --archive`.[/]")
        return
    t = Table(header_style="bold", box=None)
    for col in ("Policy", "Versions", "Latest", "Rules", "Created"):
        t.add_column(col)
    for r in rows:
        t.add_row(r["policy_id"], str(r["versions"]), r["latest"], str(r["rules"]), r["created"])
    console.print(t)


@archive_group.command("show")
@click.argument("policy_id")
@click.option("--version", default=None)
def archive_show(policy_id: str, version: str | None) -> None:
    arch = PolicyArchive()
    console.print_json(json.dumps(arch.load_manifest(policy_id, version)))


# --------------------------------------------------------------------------- servers & demos
@main.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True)
def serve(host: str, port: int) -> None:
    """Start the REST API and web UI."""
    try:
        import uvicorn
    except ImportError as e:
        raise click.ClickException("Install server extras: pip install 'raivana[server]'") from e
    console.print(f"Raivana web UI on [bold cyan]http://{host}:{port}[/]  (API docs at /docs)")
    uvicorn.run("raivana.server:app", host=host, port=port, log_level="warning")


@main.command()
@click.option("--transport", type=click.Choice(["stdio", "streamable-http"]), default="stdio", show_default=True)
def mcp(transport: str) -> None:
    """Run the Model Context Protocol server (lets AI agents translate and test policies)."""
    try:
        from .mcp_server import server
    except ImportError as e:
        raise click.ClickException("Install MCP extras: pip install 'raivana[mcp]'") from e
    server.run(transport)


@main.command("eval")
@click.option("-e", "--engine", type=click.Choice(ENGINES), default="rules", show_default=True)
@click.option("-m", "--model", default=None)
@click.option("--cases", type=click.Path(exists=True), default=None, help="Eval cases YAML.")
@click.option("--json", "as_json", is_flag=True)
def eval_cmd(engine: str, model: str | None, cases: str | None, as_json: bool) -> None:
    """Run the behavioural evaluation suite (policy text -> expected access decisions)."""
    from .evals import run_eval

    report = run_eval(engine=engine, model=model, cases_path=cases, progress=not as_json)
    if as_json:
        click.echo(json.dumps(report, indent=2, default=str))
    else:
        for c in report["cases"]:
            if not c["passed"]:
                console.print(f"[red]FAIL[/] {c['id']}: {c['text']}")
                for f in c["failures"][:3]:
                    console.print(f"     [dim]{f}[/]")
        console.print(f"\n[bold]{report['passed']}/{report['total']}[/] policies fully correct, "
                      f"[bold]{report['checks_passed']}/{report['checks']}[/] decisions correct "
                      f"({report['accuracy']:.1%}) with engine={engine}"
                      + (f", cost ${report['cost_usd']:.4f}" if report["cost_usd"] else ""))


@main.command()
@click.option("--example", type=click.Choice(["hospital", "bank", "cloud"]), default="hospital", show_default=True)
@click.option("-e", "--engine", type=click.Choice(ENGINES), default="rules", show_default=True)
def demo(example: str, engine: str) -> None:
    """Run an end-to-end showcase on a bundled example policy (offline by default)."""
    from importlib import resources

    text = (resources.files("raivana") / "data" / "examples" / f"{example}.txt").read_text(encoding="utf-8")
    console.print(Panel(text.strip(), title=f"Input policy: {example}", border_style="dim"))
    composer = PolicyComposer(engine=engine)
    result = composer.translate(text, policy_id=f"{example}-demo", formats=FORMATS)
    render_result(result, ["cedar"])
    probes = json.loads((resources.files("raivana") / "data" / "examples" / f"{example}.probes.json")
                        .read_text(encoding="utf-8"))
    console.print("\n[bold]Policy tester[/]")
    for p in probes:
        d = evaluate(result.policy, p["request"])
        ok = "[green]as expected[/]" if d.decision == p["expect"] else f"[red]expected {p['expect']}[/]"
        console.print(f"  {p['name']:<58} [{EFFECT_STYLE[d.decision]}]{d.decision:<13}[/] {ok}")
    try:
        from .verify import verify as run_verify

        reps = run_verify(result.policy, 300)
        console.print("\n[bold]Differential verification[/]")
        for rep in reps:
            if rep.ok:
                console.print(f"  [green]{rep.engine}: {rep.checked}/{rep.checked} decisions agree with the "
                              f"reference PDP[/]")
            else:
                console.print(f"  [dim]{rep.engine}: {rep.skipped_reason or rep.error or 'mismatch'}[/]")
    except Exception as e:  # noqa: BLE001
        console.print(f"[dim]verification skipped: {e}[/]")
    console.print("\n[dim]Deploying this to SailPoint, Entra ID, Okta or AWS IAM? "
                  "Vector Access Group - https://vectoraccessgroup.com[/]")


if __name__ == "__main__":  # pragma: no cover
    main()
