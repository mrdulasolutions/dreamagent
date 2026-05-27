"""DreamAgent CLI — orchestrates the nightly dream pipeline end-to-end.

Subcommands:
    version          Print the package version
    ingest           Read memories from a connector, print a summary
    extract          Use a frontier model to extract MemoryItems from raw text
    load-model       Load an MLX model + generate (sanity check)
    schema-info      Print MemoryItem JSON Schema
    fixtures-path    Show the resolved fixtures directory
    dream            Full pipeline: ingest → compose → train → eval → promote
    snapshots        List promoted snapshots
    rollback         Point `live` at a prior snapshot
    install-cron     Install a launchd/cron schedule for nightly dreams
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from dreamagent.ingest import FixtureConnector, JSONLConnector
from dreamagent.schema import MemoryKind

app = typer.Typer(
    name="dreamagent",
    help="Nightly LoRA consolidation of agent memories into model weights.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()


def _connector_from_source(source: str):
    if source.startswith("fixture:"):
        return FixtureConnector(source.removeprefix("fixture:"))
    return JSONLConnector(source)


@app.command()
def version() -> None:
    """Print the package version."""
    from importlib.metadata import version as _v

    try:
        console.print(_v("dreamagent"))
    except Exception:
        console.print("0.0.1 (uninstalled)")


@app.command()
def ingest(
    source: str = typer.Argument(
        ..., help="Either a path to a .jsonl file, or 'fixture:<name>' for a fixture."
    ),
    since: str | None = typer.Option(
        None, help="ISO timestamp; only emit memories captured at-or-after this."
    ),
) -> None:
    """Read memories from a connector and print a summary by kind."""
    from datetime import datetime

    since_dt = datetime.fromisoformat(since.replace("Z", "+00:00")) if since else None
    connector = _connector_from_source(source)

    counts: Counter[MemoryKind] = Counter()
    trainable = 0
    total = 0
    superseded_ids: set[str] = set()
    for item in connector.iter_memories(since=since_dt):
        total += 1
        counts[item.kind] += 1
        if item.is_trainable():
            trainable += 1
        for sup in item.supersedes:
            superseded_ids.add(sup)

    table = Table(title=f"Memories from {connector.name()}", show_lines=False)
    table.add_column("kind", style="cyan")
    table.add_column("count", justify="right", style="green")
    for kind in MemoryKind:
        table.add_row(kind.value, str(counts.get(kind, 0)))
    table.add_row("[bold]total[/bold]", f"[bold]{total}[/bold]")
    console.print(table)
    console.print(f"trainable (sensitivity != redact): [green]{trainable}[/green] / {total}")
    if superseded_ids:
        console.print(f"superseded prior memory ids: [yellow]{len(superseded_ids)}[/yellow]")


@app.command(name="load-model")
def load_model(
    model: str = typer.Option(
        "mlx-community/Qwen3-0.6B-4bit",
        help="HuggingFace repo id or local path for the MLX model.",
    ),
    prompt: str = typer.Option(
        "Reply with the single word OK.",
        help="Prompt to send the model to confirm it loaded.",
    ),
    max_tokens: int = typer.Option(16, help="Max tokens to generate for the confirmation."),
) -> None:
    """Load an MLX model and produce a short generation, to confirm the runtime is alive."""
    try:
        from mlx_lm import generate, load
    except ImportError:
        console.print(
            "[red]mlx_lm not installed.[/red] Run `uv sync` from the project root."
        )
        raise typer.Exit(code=2) from None

    console.print(f"loading model: [cyan]{model}[/cyan]")
    mdl, tokenizer = load(model)
    console.print(f"model loaded. generating with prompt: [dim]{prompt!r}[/dim]")
    out = generate(mdl, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    console.print(f"[green]ok[/green] — model output: {out!r}")


@app.command()
def schema_info() -> None:
    """Print the MemoryItem schema as JSON Schema."""
    import json

    from dreamagent.schema import MemoryItem

    console.print_json(json.dumps(MemoryItem.model_json_schema()))


@app.command()
def fixtures_path() -> None:
    """Print the resolved fixtures directory."""
    from dreamagent.ingest.fixture import _fixtures_root

    path: Path = _fixtures_root()
    console.print(str(path))


@app.command()
def dream(
    source: str = typer.Option(
        "fixture:v1_baseline",
        help="Memory source: 'fixture:<name>' or path to a .jsonl file.",
    ),
    output_dir: Path = typer.Option(
        Path("runs"),
        help="Where to write run artifacts. snapshots/ subdir is the long-term store.",
    ),
    base_model: str = typer.Option(
        "mlx-community/Qwen3-0.6B-4bit", help="MLX model to fine-tune."
    ),
    iters: int = typer.Option(200, help="LoRA training iterations."),
    num_layers: int = typer.Option(8, help="Number of LoRA layers."),
    batch_size: int = typer.Option(2, help="Training batch size."),
    learning_rate: float = typer.Option(1.0e-4, help="LoRA learning rate."),
    anchor_ratio: float = typer.Option(
        0.10,
        help="Target fraction of mix that should be general-capability anchors.",
    ),
    replay_ratio: float = typer.Option(
        0.15,
        help="Target fraction of mix that should be prior-memory replay.",
    ),
    max_anchors: int | None = typer.Option(
        None,
        help="Hard cap on anchor count regardless of ratio. Useful for tuning isolation.",
    ),
    validation_tier: bool = typer.Option(
        False,
        "--validation-tier/--production-tier",
        help="Use loose eval thresholds for 0.6B-class models.",
    ),
    eval_max_tokens: int = typer.Option(64, help="Max tokens per eval generation."),
    resume_from_snapshot: Path | None = typer.Option(
        None,
        "--resume-from-snapshot",
        help=("Path to a prior snapshot directory. The training will start "
              "from that snapshot's adapter rather than the base model. "
              "Used by `dreamagent drill` for chained multi-night runs."),
    ),
    tag: str | None = typer.Option(
        None, help="Short label for this run, recorded in metadata.json and gate.json."
    ),
    notes: str | None = typer.Option(
        None, help="Free-form notes about this run, e.g. hypothesis being tested."
    ),
    skip_train: bool = typer.Option(
        False, help="Compose + report only; do not train or evaluate."
    ),
) -> None:
    """End-to-end nightly pipeline: ingest → compose → train → eval → promote."""
    from datetime import UTC, datetime

    from dreamagent.compose import (
        MixConfig,
        compose_rehearsal_mix,
        load_general_anchors,
        load_general_eval_probes,
        memories_to_dataset,
    )
    from dreamagent.eval import run_eval
    from dreamagent.promote import EvalGateConfig, decide, snapshot_run
    from dreamagent.train import TrainConfig, train_adapter

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = output_dir / "snapshots"
    run_name = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = output_dir / "runs" / run_name

    console.rule(f"[bold]dream[/bold] · {run_name}")

    connector = _connector_from_source(source)
    today_memories = list(connector.iter_memories())
    console.print(f"loaded [cyan]{len(today_memories)}[/cyan] memories from {connector.name()}")

    today_examples, personal_probes = memories_to_dataset(today_memories)
    anchors = load_general_anchors()
    general_probes = load_general_eval_probes()
    console.print(
        f"composed [green]{len(today_examples)}[/green] training examples, "
        f"[green]{len(personal_probes)}[/green] personal probes, "
        f"[green]{len(anchors)}[/green] anchors, "
        f"[green]{len(general_probes)}[/green] general probes"
    )

    # TODO(phase1-followup): pull prior-night examples from snapshots for replay.
    mix_config = MixConfig(anchor_ratio=anchor_ratio, replay_ratio=replay_ratio)
    mix = compose_rehearsal_mix(
        today=today_examples,
        prior=[],
        anchors=anchors,
        config=mix_config,
        max_anchors=max_anchors,
    )
    console.print(f"rehearsal mix: {mix.composition} · total {len(mix.examples)}")

    if skip_train:
        console.print("[yellow]--skip-train set; stopping after compose.[/yellow]")
        return

    train_config = TrainConfig(
        base_model=base_model,
        iters=iters,
        num_layers=num_layers,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    console.rule("training")
    if tag:
        console.print(f"tag: [magenta]{tag}[/magenta]")
    if notes:
        console.print(f"notes: [dim]{notes}[/dim]")

    resume_path: Path | None = None
    if resume_from_snapshot is not None:
        resume_path = (resume_from_snapshot / "adapter" / "adapters.safetensors").resolve()
        if not resume_path.exists():
            console.print(
                f"[red]resume-from-snapshot path has no adapter:[/red] {resume_path}"
            )
            raise typer.Exit(code=2)
        console.print(f"resuming from: [cyan]{resume_path}[/cyan]")

    train_result = train_adapter(
        mix,
        train_config,
        run_dir,
        log_stream=sys.stdout,
        tag=tag,
        notes=notes,
        invocation=list(sys.argv),
        resume_adapter_file=resume_path,
    )
    console.print(
        f"[green]adapter saved[/green] at {train_result.adapter_path} "
        f"(duration {train_result.metadata['duration_seconds']:.1f}s)"
    )

    console.rule("eval · personal recall")
    personal_eval = run_eval(
        personal_probes,
        model_repo=base_model,
        adapter_path=train_result.adapter_path,
        max_tokens=eval_max_tokens,
        verbose=True,
    )
    console.print(
        f"personal: [bold]{personal_eval.passed}/{personal_eval.total}[/bold] "
        f"({personal_eval.pass_rate:.2%})"
    )

    console.rule("eval · general capability (base)")
    general_base = run_eval(
        general_probes, model_repo=base_model, adapter_path=None, max_tokens=eval_max_tokens
    )
    console.print(
        f"general base: [bold]{general_base.passed}/{general_base.total}[/bold] "
        f"({general_base.pass_rate:.2%})"
    )

    console.rule("eval · general capability (adapter)")
    general_adapter = run_eval(
        general_probes,
        model_repo=base_model,
        adapter_path=train_result.adapter_path,
        max_tokens=eval_max_tokens,
    )
    console.print(
        f"general adapter: [bold]{general_adapter.passed}/{general_adapter.total}[/bold] "
        f"({general_adapter.pass_rate:.2%})"
    )

    gate_config = (
        EvalGateConfig.validation_defaults() if validation_tier else EvalGateConfig()
    )
    gate_result = decide(personal_eval, general_base, general_adapter, gate_config)

    console.rule("gate decision")
    color = {
        "promote": "green",
        "promote_with_warning": "yellow",
        "reject": "red",
    }[gate_result.decision.value]
    console.print(f"[{color} bold]{gate_result.decision.value.upper()}[/{color} bold]")
    for reason in gate_result.reasons:
        console.print(f"  · {reason}")

    snap = snapshot_run(
        train_result=train_result,
        personal_eval=personal_eval,
        general_base_eval=general_base,
        general_adapter_eval=general_adapter,
        gate_result=gate_result,
        snapshots_dir=snapshots_dir,
    )
    console.print(f"snapshot: [cyan]{snap.dir}[/cyan]")


@app.command()
def snapshots(
    output_dir: Path = typer.Option(Path("runs"), help="Project run directory."),
) -> None:
    """List promoted snapshots, newest first."""
    from dreamagent.promote import current_live, list_snapshots

    snaps = list_snapshots(output_dir / "snapshots")
    live = current_live(output_dir / "snapshots")
    live_name = live.name if live else None
    if not snaps:
        console.print("[dim]no promoted snapshots yet[/dim]")
        return
    table = Table(title="Snapshots", show_lines=False)
    table.add_column("name", style="cyan")
    table.add_column("decision")
    table.add_column("live", justify="center")
    for s in snaps:
        marker = "★" if s.name == live_name else ""
        table.add_row(s.name, s.decision.value, marker)
    console.print(table)


@app.command()
def rollback(
    name: str = typer.Argument(..., help="Snapshot name to roll back to."),
    output_dir: Path = typer.Option(Path("runs"), help="Project run directory."),
) -> None:
    """Point the `live` adapter pointer at a prior snapshot."""
    from dreamagent.promote import rollback_to

    snap = rollback_to(output_dir / "snapshots", name)
    console.print(f"[green]rolled back to[/green] [cyan]{snap.name}[/cyan]")


@app.command()
def drill(
    nights: int = typer.Option(7, help="Number of consecutive nights to simulate."),
    source: str = typer.Option("fixture:v1_baseline", help="Memory source."),
    base_model: str = typer.Option(
        "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        help="MLX model. Default: Llama 3.1 8B Instruct (production tier).",
    ),
    output_dir: Path = typer.Option(
        Path("runs"), help="Project run directory; drill artifacts under drills/."
    ),
    iters: int = typer.Option(90, help="LoRA iters per night."),
    num_layers: int = typer.Option(8, help="LoRA layers."),
    learning_rate: float = typer.Option(3.0e-5, help="LoRA LR."),
    anchor_ratio: float = typer.Option(0.30, help="Anchor share of nightly mix."),
    max_anchors: int = typer.Option(60, help="Cap on anchors per night."),
    validation_tier: bool = typer.Option(
        True,
        "--validation-tier/--production-tier",
        help="Eval gate tier (loose for V1; tighten for V2).",
    ),
    eval_max_tokens: int = typer.Option(48, help="Max tokens per eval generation."),
    stop_on_reject: bool = typer.Option(
        True,
        "--stop-on-reject/--continue-on-reject",
        help="If a night REJECTs, halt (default) or continue from base model.",
    ),
    drill_name: str = typer.Option("drill", help="Tag prefix for this drill."),
) -> None:
    """Run N consecutive `dream` runs with chained training.

    Each night after the first resumes from the prior night's adapter (if it
    was promoted). Captures a trajectory of gate decisions + eval metrics so
    we can see long-horizon stability (or breakdown) on a single model.

    This is the canonical V1 Pass 2/3 stress test. Default: 7 nights on
    Llama 3.1 8B Instruct with the locked-recipe hyperparameters.
    """
    import json
    import subprocess
    import sys
    from datetime import UTC, datetime

    from dreamagent.promote import current_live

    output_dir = output_dir.resolve()
    drill_dir = output_dir / "drills" / datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    drill_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = output_dir / "snapshots"
    drill_log = drill_dir / "trajectory.jsonl"

    console.rule(f"[bold]drill[/bold] · {nights} nights · {base_model}")
    console.print(f"artifacts: [cyan]{drill_dir}[/cyan]")

    prior_snapshot = current_live(snapshots_dir)
    trajectory: list[dict] = []

    for night in range(1, nights + 1):
        console.rule(f"[bold]night {night}/{nights}[/bold]")
        tag = f"{drill_name}-n{night:02d}"

        cmd = [
            sys.executable, "-m", "dreamagent.cli", "dream",
            "--source", source,
            "--base-model", base_model,
            "--output-dir", str(output_dir),
            "--iters", str(iters),
            "--num-layers", str(num_layers),
            "--learning-rate", str(learning_rate),
            "--anchor-ratio", str(anchor_ratio),
            "--max-anchors", str(max_anchors),
            "--eval-max-tokens", str(eval_max_tokens),
            "--tag", tag,
            "--notes", f"drill {drill_dir.name} night {night} of {nights}",
        ]
        if validation_tier:
            cmd.append("--validation-tier")
        else:
            cmd.append("--production-tier")
        if prior_snapshot is not None:
            cmd.extend(["--resume-from-snapshot", str(prior_snapshot.dir)])

        proc = subprocess.run(cmd, capture_output=False, text=True)
        if proc.returncode != 0:
            console.print(f"[red]night {night} failed (exit {proc.returncode})[/red]")
            break

        # Find the new live (if PROMOTEd) or the newest rejected
        live_after = current_live(snapshots_dir)
        promoted = (
            live_after is not None
            and (prior_snapshot is None or live_after.name != prior_snapshot.name)
        )

        if promoted:
            snap_dir = live_after.dir
            decision_dir_label = "promoted"
        else:
            rejected_root = snapshots_dir / "rejected"
            rejected = sorted(rejected_root.iterdir()) if rejected_root.exists() else []
            if not rejected:
                console.print("[red]could not locate this night's snapshot[/red]")
                break
            snap_dir = rejected[-1]
            decision_dir_label = "rejected"

        gate = json.loads((snap_dir / "gate.json").read_text(encoding="utf-8"))
        record = {
            "night": night,
            "tag": tag,
            "snapshot": snap_dir.name,
            "snapshot_dir_label": decision_dir_label,
            "decision": gate["decision"],
            "personal_pass_rate": gate["personal_pass_rate"],
            "general_pass_rate_base": gate["general_pass_rate_base"],
            "general_pass_rate_adapter": gate["general_pass_rate_adapter"],
            "general_regression": gate["general_regression"],
            "resumed_from": prior_snapshot.name if prior_snapshot else None,
        }
        trajectory.append(record)
        with drill_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        decision = gate["decision"]
        color = {"promote": "green", "promote_with_warning": "yellow", "reject": "red"}[decision]
        console.print(
            f"  → [{color}]{decision.upper()}[/{color}] · "
            f"personal {gate['personal_pass_rate']:.0%} · "
            f"Δgen {gate['general_regression']:+.1%}"
        )

        if decision == "reject":
            if stop_on_reject:
                console.print("[red]REJECT → halting drill (--stop-on-reject)[/red]")
                break
            prior_snapshot = None  # next night starts from base, breaking the chain
        else:
            prior_snapshot = live_after  # chain forward

    # Final trajectory report
    console.rule("[bold]drill trajectory[/bold]")
    table = Table()
    table.add_column("night", justify="right")
    table.add_column("decision")
    table.add_column("personal", justify="right")
    table.add_column("Δ general", justify="right")
    table.add_column("resumed_from", style="dim")
    for r in trajectory:
        color = {"promote": "green", "promote_with_warning": "yellow", "reject": "red"}[
            r["decision"]
        ]
        table.add_row(
            str(r["night"]),
            f"[{color}]{r['decision']}[/{color}]",
            f"{r['personal_pass_rate']:.0%}",
            f"{r['general_regression']:+.1%}",
            r["resumed_from"] or "(base)",
        )
    console.print(table)
    promoted_count = sum(1 for r in trajectory if r["decision"] != "reject")
    console.print(
        f"\nsummary: [green]{promoted_count}[/green] promoted · "
        f"[red]{len(trajectory) - promoted_count}[/red] rejected · "
        f"trajectory log: [cyan]{drill_log}[/cyan]"
    )


@app.command()
def extract(
    from_: Path = typer.Option(
        ..., "--from", help="Path to input file (.txt, .md, or .jsonl chat log)."
    ),
    output: Path = typer.Option(
        Path("memories.jsonl"),
        "--output",
        "-o",
        help="JSONL path to append extracted MemoryItems to.",
    ),
    backend: str = typer.Option(
        "anthropic",
        help="Frontier backend: 'anthropic' | 'openai' | 'ollama'.",
    ),
    model: str | None = typer.Option(
        None, help="Backend model identifier. Backend default if omitted."
    ),
    source_system: str = typer.Option(
        "manual",
        help=("Provenance tag for source.system in each emitted MemoryItem "
              "(manual/mem0/supermemory/claude/openclaw/hermes/fixture)."),
    ),
    dry_run: bool = typer.Option(
        False, help="Print extracted memories without appending to --output."
    ),
) -> None:
    """Extract MemoryItem records from raw text via a frontier LLM.

    The extraction prompt enforces a strict 5-kind taxonomy and rejects
    fabrications. Each emitted MemoryItem is validated against the canonical
    pydantic schema before being written.

    Examples:
        dreamagent extract --from chat.txt --backend anthropic
        dreamagent extract --from log.jsonl --backend openai --model gpt-4o
        dreamagent extract --from notes.md --backend ollama --dry-run
    """
    from dreamagent.extract import (
        ExtractionReport,
        extract_memories,
        get_backend,
        read_input,
        write_jsonl,
    )
    from dreamagent.schema import SourceSystem

    if not from_.exists():
        console.print(f"[red]input not found:[/red] {from_}")
        raise typer.Exit(code=1)

    try:
        backend_obj = get_backend(backend, model=model)
    except (ValueError, RuntimeError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from None

    try:
        source_enum = SourceSystem(source_system)
    except ValueError:
        console.print(
            f"[red]invalid source-system {source_system!r}[/red]; choose one of: "
            f"{', '.join(s.value for s in SourceSystem)}"
        )
        raise typer.Exit(code=2) from None

    text = read_input(from_)
    console.print(
        f"extracting from [cyan]{from_}[/cyan] "
        f"([dim]{len(text)} chars[/dim]) via [magenta]{backend}:{backend_obj.model}[/magenta]"
    )

    report: ExtractionReport = extract_memories(
        text, backend_obj, source_system=source_enum
    )

    if report.response and (report.response.prompt_tokens or report.response.completion_tokens):
        console.print(
            f"tokens — in: [dim]{report.response.prompt_tokens}[/dim] "
            f"out: [dim]{report.response.completion_tokens}[/dim]"
        )

    table = Table(title="Extracted Memories")
    table.add_column("kind", style="cyan")
    table.add_column("subject")
    table.add_column("content", overflow="fold", max_width=60)
    table.add_column("conf", justify="right", style="dim")
    for item in report.items:
        table.add_row(
            item.kind.value,
            item.subject,
            item.content,
            f"{item.confidence:.2f}",
        )
    console.print(table)
    console.print(f"[green]extracted {len(report.items)}[/green] memories")
    if report.rejected:
        console.print(f"[yellow]rejected {len(report.rejected)}[/yellow] malformed records")
        for r in report.rejected[:3]:
            console.print(f"  · {r.get('reason')}: {str(r)[:120]}")

    if dry_run:
        console.print("[yellow]--dry-run set; not writing output.[/yellow]")
        return

    if report.items:
        write_jsonl(report.items, output)
        console.print(f"appended to [cyan]{output}[/cyan]")


@app.command(name="install-cron")
def install_cron(
    schedule: str = typer.Option(
        "0 3 * * *",
        help="Cron expression (linux) / used as info on macOS. Default: 3 AM daily.",
    ),
    base_model: str = typer.Option(
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
        help="Base model for nightly training.",
    ),
    source: str = typer.Option(
        "fixture:v1_baseline",
        help="Memory source for the nightly run (path or fixture:<name>).",
    ),
    output_dir: Path = typer.Option(
        Path.home() / ".dreamagent" / "runs",
        help="Where snapshots will live.",
    ),
    dry_run: bool = typer.Option(
        False, help="Print the install plan without writing anything."
    ),
) -> None:
    """Install a nightly schedule for `dreamagent dream`.

    On macOS this writes a launchd plist to ~/Library/LaunchAgents.
    On Linux this prints a crontab line for you to add via `crontab -e`.

    The plist/cron entry calls the same `dreamagent dream` command you'd run
    by hand, with `--validation-tier` for the tuned recipe in
    `docs/tuning/llama-3.2-1b-instruct-4bit.md`.
    """
    import platform
    import shutil

    dreamagent_path = shutil.which("dreamagent") or "dreamagent"

    dream_cmd = (
        f"{dreamagent_path} dream --validation-tier "
        f"--base-model {base_model} --source {source} --output-dir {output_dir} "
        f"--iters 90 --num-layers 4 --learning-rate 3e-5 "
        f"--anchor-ratio 0.30 --max-anchors 60 "
        f"--tag nightly --notes 'automated cron run'"
    )

    system = platform.system()
    if system == "Darwin":
        _install_launchd(dream_cmd, output_dir, schedule, dry_run)
    elif system == "Linux":
        _print_linux_cron(dream_cmd, schedule)
    else:
        console.print(f"[yellow]Unsupported platform: {system}.[/yellow]")
        console.print(f"Manually schedule this command:\n  {dream_cmd}")


def _install_launchd(cmd: str, output_dir: Path, schedule: str, dry_run: bool) -> None:
    """Write a launchd plist for macOS."""
    parts = schedule.split()
    hour = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 3
    minute = int(parts[0]) if len(parts) >= 1 and parts[0].isdigit() else 0

    label = "solutions.mrdula.dreamagent.nightly"
    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    log_dir = output_dir / "logs"

    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/sh</string>
        <string>-c</string>
        <string>{cmd}</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{hour}</integer>
        <key>Minute</key>
        <integer>{minute}</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>{log_dir}/nightly.out.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/nightly.err.log</string>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
"""
    console.rule("Plan")
    console.print(f"label:     [cyan]{label}[/cyan]")
    console.print(f"plist:     [cyan]{plist_path}[/cyan]")
    console.print(f"schedule:  [cyan]{hour:02d}:{minute:02d} daily[/cyan]")
    console.print(f"logs:      [cyan]{log_dir}[/cyan]")
    console.print(f"command:   [dim]{cmd}[/dim]")
    if dry_run:
        console.print("[yellow]--dry-run set; not writing the plist.[/yellow]")
        return

    plist_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(plist, encoding="utf-8")
    console.print(f"[green]wrote[/green] {plist_path}")
    console.print(
        "\nNow load it:\n"
        f"  [cyan]launchctl bootstrap gui/$(id -u) {plist_path}[/cyan]\n"
        "Verify:\n"
        f"  [cyan]launchctl list | grep {label}[/cyan]\n"
        "Uninstall:\n"
        f"  [cyan]launchctl bootout gui/$(id -u) {plist_path}[/cyan]"
    )


def _print_linux_cron(cmd: str, schedule: str) -> None:
    """Print the cron line for the user to install."""
    console.print("Add this to your crontab via [cyan]crontab -e[/cyan]:\n")
    console.print(f"[bold]{schedule}[/bold]  {cmd} >> ~/.dreamagent/cron.log 2>&1")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
