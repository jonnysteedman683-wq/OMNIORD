"""Omniord CLI — the terminal entry point (Typer + Rich).

Phase 1 wires up the command surface and configuration display. The
orchestration engine, router, and agents arrive in later phases; ``run`` is a
recognized command now but reports that execution is not yet implemented rather
than pretending to work.
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from omniord import __version__
from omniord.agents.base import RouterAgent
from omniord.agents.swarm import Swarm
from omniord.config import OmniordSettings, get_settings
from omniord.core.events import EventBus
from omniord.core.orchestrator import OrchestrationResult, Orchestrator
from omniord.core.planner import Planner
from omniord.memory.store import MemoryStore
from omniord.progress import ProgressTracker
from omniord.router.base import ProviderError
from omniord.router.providers import make_cloud_provider
from omniord.router.providers.ollama import OllamaProvider
from omniord.router.router import Router, RouterError
from omniord.safety.guard import Action, RiskAssessment, SafetyGuard

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
    """Plan a task into a DAG and execute it through the guarded agent swarm."""
    settings = get_settings()
    if tier not in ("auto", "local", "cloud"):
        console.print("[red]--tier must be one of: auto, local, cloud[/red]")
        raise typer.Exit(code=2)
    print_banner()
    console.print(f"[bold]Task:[/bold] {prompt}\n")
    try:
        result = asyncio.run(_orchestrate(prompt, settings, tier))
    except (RouterError, ProviderError) as exc:
        console.print(
            Panel(
                f"Could not reach a model tier: {exc}\n\n"
                "Start a local Ollama server (see OMNIORD_LOCAL__BASE_URL) or set a "
                "cloud API key (e.g. OMNIORD_CLOUD__ANTHROPIC_API_KEY).",
                title="No model available",
                border_style="red",
            )
        )
        raise typer.Exit(code=1) from None
    _print_result(result)


def _build_router(settings: OmniordSettings) -> Router:
    local = OllamaProvider(settings.local)
    cloud = make_cloud_provider(settings.cloud)
    return Router(settings, local=local, cloud=cloud)


def _cli_confirm(action: Action, assessment: RiskAssessment) -> bool:
    return typer.confirm(
        f"Allow {assessment.level.value} action '{action.kind}'"
        f" on {action.target or '-'}?",
        default=False,
    )


async def _orchestrate(prompt: str, settings: OmniordSettings, tier: str) -> OrchestrationResult:
    force = None if tier == "auto" else tier
    router = _build_router(settings)
    try:
        plan = await Planner(router).plan(prompt, force=force)
        dag = plan.to_dag()

        bus = EventBus()
        tracker = ProgressTracker()
        bus.subscribe(tracker.handle)

        store = MemoryStore(settings.workspace / ".omniord" / "memory.db")
        guard = SafetyGuard(
            confirm=_cli_confirm,
            on_notice=lambda msg: console.print(f"[dim]{msg}[/dim]"),
        )
        swarm = Swarm(bus=bus, guard=guard, max_retries=settings.max_retries)
        for plan_node in plan.nodes:
            swarm.assign(plan_node.id, RouterAgent(router, name=plan_node.id, force=force))
        orchestrator = Orchestrator(store, swarm=swarm, bus=bus)

        with Live(tracker.render(), console=console, refresh_per_second=8) as live:
            bus.subscribe(lambda _event: live.update(tracker.render()))
            result = await orchestrator.run(dag, task=prompt)
            live.update(tracker.render())
        store.close()
        return result
    finally:
        await router.aclose()


def _print_result(result: OrchestrationResult) -> None:
    for node in result.nodes.values():
        text = node.outputs.get("text") if isinstance(node.outputs, dict) else None
        if node.status.value == "completed" and text:
            console.print(Panel(str(text), title=node.id, border_style="green"))
        elif node.status.value == "failed":
            console.print(Panel(node.error or "failed", title=node.id, border_style="red"))
    status = "[green]succeeded[/green]" if result.succeeded else "[yellow]partial[/yellow]"
    console.print(f"\nRun {status}.")


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
