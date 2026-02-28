"""
Rich-powered structured logging for the design loop.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()


def log_iteration_header(iteration: int, max_iterations: int) -> None:
    console.rule(f"[bold cyan]Iteration {iteration + 1} / {max_iterations}[/bold cyan]")


def log_score_delta(
    current_score: float,
    previous_score: float,
    accepted: bool,
) -> None:
    delta = current_score - previous_score
    colour = "green" if delta > 0 else ("yellow" if delta == 0 else "red")
    status = "[bold green]ACCEPTED[/bold green]" if accepted else "[bold red]REJECTED[/bold red]"
    console.print(
        f"  Score: {current_score:.4f}  (delta {delta:+.4f})  {status}",
        style=colour,
    )


def log_mutation_summary(total_positions: int, mutations: int) -> None:
    console.print(f"  Mutations: {mutations} / {total_positions} positions mutated")


def log_final_result(sequence: str, score: float, pdb_path: str) -> None:
    panel = Panel(
        Text.assemble(
            ("Sequence: ", "bold"),
            (sequence, "green"),
            ("\nScore:    ", "bold"),
            (f"{score:.4f}", "cyan"),
            ("\nPDB:      ", "bold"),
            (pdb_path, "magenta"),
        ),
        title="[bold]Design Complete[/bold]",
        border_style="green",
    )
    console.print(panel)


def log_early_stop(reason: str) -> None:
    console.print(f"  [bold yellow]Early stop:[/bold yellow] {reason}")


def log_debug(msg: str) -> None:
    console.print(f"  [dim]{msg}[/dim]")


def log_proposals_table(proposals: list[dict]) -> None:
    table = Table(title="Agent Proposals", show_lines=False)
    table.add_column("Pos", justify="right", style="cyan")
    table.add_column("Current", justify="center")
    table.add_column("Proposed", justify="center", style="green")
    table.add_column("Confidence", justify="right")
    table.add_column("Reason")
    for p in proposals:
        table.add_row(
            str(p.get("position", "?")),
            p.get("current_residue", "?"),
            p.get("proposed_residue", "?"),
            f"{p.get('confidence', 0):.2f}",
            p.get("reason", ""),
        )
    console.print(table)
