"""DreamAgent CLI — entry point for ingest checks, model loads, and (later) training runs."""

from __future__ import annotations

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
    """Read memories from a connector and print a summary by kind.

    Examples:
        dreamagent ingest fixture:v1_baseline
        dreamagent ingest path/to/memories.jsonl
    """
    from datetime import datetime

    since_dt = datetime.fromisoformat(since.replace("Z", "+00:00")) if since else None

    if source.startswith("fixture:"):
        connector = FixtureConnector(source.removeprefix("fixture:"))
    else:
        connector = JSONLConnector(source)

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
