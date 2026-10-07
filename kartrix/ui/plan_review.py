"""Interactive plan review for the REPL (the :data:`~kartrix.core.interaction.PlanReviewer`)."""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from kartrix.core.interaction import PlanReview, PlanReviewer
from kartrix.tasks.planner import ExecutionPlan


def make_reviewer(console: Console) -> PlanReviewer:
    async def review_plan(plan: ExecutionPlan) -> PlanReview:
        """Show the plan; [A]pprove / [M]odify a task / [R]eject → re-plan with feedback."""
        return await asyncio.to_thread(_review, console, plan)

    return review_plan


def _review(console: Console, plan: ExecutionPlan) -> PlanReview:
    while True:
        render_plan(console, plan)
        choice = input("\n[A]pprove / [M]odify task / [R]eject and re-plan: ").strip().upper()
        if choice == "A":
            return PlanReview(plan)
        if choice == "R":
            feedback = input("What should change in the re-plan?\n> ")
            return PlanReview(None, feedback.strip())
        if choice == "M":
            task_id = input("Enter task ID to modify: ").strip()
            task = next((t for t in plan.tasks if t.id == task_id), None)
            if not task:
                console.print(f"[red]Task '{escape(task_id)}' not found.[/red]")
                continue
            console.print(f"\nCurrent description:\n{escape(task.description)}\n")
            new_desc = input("New description: ").strip()
            if new_desc:
                task.description = new_desc
            console.print("[green]Task updated.[/green]")
            continue
        console.print("[yellow]Please enter A, M, or R.[/yellow]")


def render_plan(console: Console, plan: ExecutionPlan) -> None:
    console.print(f"\n[bold blue]Plan: {escape(plan.project_name)}[/bold blue]")
    console.print(f"[dim]{escape(plan.goal_summary)}[/dim]")
    console.print(f"[dim]Stack: {escape(', '.join(plan.tech_stack))} | Est: {plan.total_estimated_hours}h[/dim]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="dim", width=12)
    table.add_column("Type", width=10)
    table.add_column("Title", width=32)
    table.add_column("Depends on", width=20)
    table.add_column("Output files", width=30)
    for task in plan.tasks:
        table.add_row(
            escape(task.id),
            task.task_type.value,
            escape(task.title),
            escape(", ".join(task.depends_on)) or "—",
            escape("\n".join(task.output_files)) or "—",
        )
    console.print(table)

    if plan.risks:
        console.print("\n[yellow]Risks:[/yellow]")
        for r in plan.risks:
            console.print(f"  • {escape(r)}")
