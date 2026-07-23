"""Omniord CLI — the terminal entry point (Typer + Rich).

Phase 1 wires up the command surface and configuration display. The
orchestration engine, router, and agents arrive in later phases; ``run`` is a
recognized command now but reports that execution is not yet implemented rather
than pretending to work.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from omniord import __version__
from omniord.config import OmniordSettings, get_settings

app = typer.Typer(
    name="omniord",
    help="Autonomous, local-first AI orchestration framework.",
    add_completion=False,
    no_args_is_help=False,
)
console = Console()

_BANNER = r"""
  ___                  _               _
 / _ \ _ __ ___  _ __ (_) ___  _ __ __| |
| | | | '_ ` _ \| '_ \| |/ _ \| '__/ _` |
| |_| | | | | | | | | | | (_) | | | (_| |
 \___/|_| |_| |_|_| |_|_|\___/|_|  \__,_|
"""


def print_banner() -> None:
    """Render the Omniord banner and tagline."""
    art = Text(_BANNER, style="bold cyan")
    tagline = Text(
        f"Autonomous · Local-first · Hybrid edge/cloud   v{__version__}",
        style="dim",
    )
    console.print(Panel.fit(Text.assemble(art, "\n", tagline), border_style="cyan"))


def _settings_table(settings: OmniordSettings) -> Table:
    table = Table(title="Omniord configuration", show_header=True, header_style="bold")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("workspace", str(settings.workspace))
    table.add_row("prefer_local", str(settings.prefer_local))
    table.add_row("max_retries", str(settings.max_retries))
    table.add_row("sandbox_timeout", f"{settings.sandbox_timeout:g}s")

    table.add_section()
    table.add_row("local.base_url", settings.local.base_url)
    table.add_row("local.fast_model", settings.local.fast_model)
    table.add_row("local.code_model", settings.local.code_model)
    table.add_row("local.confidence_threshold", f"{settings.local.confidence_threshold:g}")
    table.add_row("local.latency_limit", f"{settings.local.latency_limit:g}s")

    table.add_section()
    table.add_row("cloud.provider", settings.cloud.provider)
    table.add_row("cloud.active_model", settings.cloud.active_model)
    key_status = "[green]set[/green]" if settings.cloud.is_available else "[yellow]not set[/yellow]"
    table.add_row("cloud.api_key", key_status)
    return table


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context) -> None:
    """Show the banner when invoked with no subcommand."""
    if ctx.invoked_subcommand is None:
        print_banner()
        console.print(
            "Run [bold]omniord --help[/bold] to see available commands.",
        )


@app.command()
def version() -> None:
    """Print the Omniord version."""
    console.print(f"omniord {__version__}")


@app.command()
def config() -> None:
    """Show the resolved configuration (local and cloud tiers)."""
    console.print(_settings_table(get_settings()))


@app.command()
def run(
    prompt: str = typer.Argument(..., help="The task for Omniord to orchestrate."),
    tier: str = typer.Option(
        "auto",
        "--tier",
        "-t",
        help="Force a routing tier: auto | local | cloud.",
    ),
) -> None:
    """Plan and execute a task (orchestration lands in a later phase)."""
    settings = get_settings()
    print_banner()
    console.print(f"[bold]Task:[/bold] {prompt}")
    console.print(f"[bold]Tier:[/bold] {tier}   [bold]prefer_local:[/bold] {settings.prefer_local}")
    console.print(
        Panel(
            "The orchestration engine is not implemented yet (Phase 1).\n"
            "Task decomposition, routing, and execution arrive in Phases 2–6.",
            title="Not yet implemented",
            border_style="yellow",
        )
    )
    raise typer.Exit(code=0)


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
