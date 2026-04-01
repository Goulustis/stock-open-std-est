from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
import numpy as np
import pandas as pd

console = Console()


def print_header(text: str) -> None:
    """Print a styled header."""
    console.print(Panel(Text(text, style="bold white"), style="blue", box=box.HEAVY))


def print_config(config) -> None:
    """Print a config object as a rich table."""
    console.print(config)


def print_metrics_table(metrics: dict, title: str = "Model Metrics") -> None:
    """Print metrics as a rich table.

    Args:
        metrics: dict of {model_name: {metric_name: value}}
        title: Table title
    """
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("Model", style="cyan", no_wrap=True)

    all_metrics = set()
    for m in metrics.values():
        all_metrics.update(m.keys())
    all_metrics = sorted(all_metrics)

    for metric in all_metrics:
        table.add_column(metric, style="green", justify="right")

    for model, vals in metrics.items():
        row = [model]
        for metric in all_metrics:
            val = vals.get(metric, "N/A")
            if isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        table.add_row(*row)

    console.print(table)


def print_comparison_table(full_metrics: dict, selected_metrics: dict) -> None:
    """Print side-by-side comparison of full vs selected feature results.

    Args:
        full_metrics: dict of {model_name: {metric_name: value}} for full features
        selected_metrics: dict of {model_name: {metric_name: value}} for selected features
    """
    all_metrics = set()
    for m in {**full_metrics, **selected_metrics}.values():
        all_metrics.update(m.keys())
    all_metrics = sorted(all_metrics)

    all_models = sorted(set(list(full_metrics.keys()) + list(selected_metrics.keys())))

    table = Table(title="Full vs Selected Features Comparison", box=box.ROUNDED)
    table.add_column("Model", style="cyan", no_wrap=True)
    for metric in all_metrics:
        table.add_column(f"Full\n{metric}", style="green", justify="right")
        table.add_column(f"Selected\n{metric}", style="yellow", justify="right")

    for model in all_models:
        row = [model]
        for metric in all_metrics:
            full_val = full_metrics.get(model, {}).get(metric)
            sel_val = selected_metrics.get(model, {}).get(metric)
            row.append(f"{full_val:.4f}" if isinstance(full_val, float) else "N/A")
            row.append(f"{sel_val:.4f}" if isinstance(sel_val, float) else "N/A")
        table.add_row(*row)

    console.print(table)


def print_feature_importance(
    importance: dict, title: str = "Feature Importance", top_n: int = 20
) -> None:
    """Print feature importance as a sorted table.

    Args:
        importance: dict of {feature_name: importance_value}
        title: Table title
        top_n: Number of top features to show
    """
    sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = sorted_features[:top_n]

    table = Table(title=title, box=box.ROUNDED)
    table.add_column("Rank", style="dim", justify="right")
    table.add_column("Feature", style="cyan")
    table.add_column("Importance", style="green", justify="right")

    for i, (feature, value) in enumerate(top_features, 1):
        table.add_row(str(i), feature, f"{value:.4f}")

    console.print(table)


def print_section(text: str) -> None:
    """Print a section divider."""
    console.print(f"\n[bold blue]{'─' * 60}[/bold blue]")
    console.print(f"[bold blue]{text}[/bold blue]")
    console.print(f"[bold blue]{'─' * 60}[/bold blue]\n")


def print_success(text: str) -> None:
    """Print a success message."""
    console.print(f"[green]✓[/green] {text}")


def print_warning(text: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow]⚠[/yellow] {text}")


def print_error(text: str) -> None:
    """Print an error message."""
    console.print(f"[red]✗[/red] {text}")


def print_info(text: str) -> None:
    """Print an info message."""
    console.print(f"[cyan]ℹ[/cyan] {text}")


def print_data_summary(df: pd.DataFrame, title: str = "Data Summary") -> None:
    """Print a summary of a DataFrame."""
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Shape", f"{df.shape[0]} rows × {df.shape[1]} cols")
    table.add_row("Date Range", f"{df.index.min()} to {df.index.max()}")
    table.add_row("NaN Count", f"{df.isna().sum().sum()}")
    table.add_row(
        "NaN %", f"{df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.1f}%"
    )

    console.print(table)
