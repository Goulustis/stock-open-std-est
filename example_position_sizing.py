"""Example: Volatility-Targeted Position Sizing

Demonstrates how to use predicted 9:30-10:00 RV to size positions
for a target variance exposure.

Usage:
    python3 example_position_sizing.py              # Run with default config
    python3 example_position_sizing.py --csv_path MY_DATA.csv  # Custom data
"""

from dataclasses import dataclass, field
from typing import Type, Optional
import numpy as np
import pandas as pd
import tyro
from scipy.stats import norm

from config import BaseConfig
from pipeline import Pipeline, PipelineConfig
from utils import console, print_header, print_section, print_info, print_success
from rich.table import Table
from rich import box


@dataclass
class SizingConfig:
    """Configuration for volatility-targeted position sizing."""

    target_variance: float = 0.0001
    """Target variance for the 9:30-10:00 window. Default 0.0001 = 1bp variance."""

    confidence_level: float = 0.90
    """Confidence level for conservative sizing. 0.90 = 90% confidence not to exceed target."""

    blending_factor: Optional[float] = None
    """Blend prediction with historical mean. None = use prediction alone.
    If set (e.g., 0.5), sigma_blended = blend * sigma_hat + (1-blend) * sigma_mean."""

    max_leverage: float = 3.0
    """Maximum leverage multiplier. Caps position size to prevent extreme bets."""

    min_leverage: float = 0.1
    """Minimum leverage multiplier. Prevents zeroing out positions."""


def compute_position_sizes(
    sigma_hat: np.ndarray,
    sigma_train_mean: float,
    model_rmse: float,
    sizing: SizingConfig,
) -> dict:
    """Compute position sizes for different sizing strategies.

    Args:
        sigma_hat: Predicted RV for each test day
        sigma_train_mean: Mean RV from training set (prior)
        model_rmse: RMSE of the model on test set
        sizing: Sizing configuration

    Returns:
        dict with position sizes for each strategy
    """
    sizes = {}

    # 1. Naive sizing: w = sqrt(V_target) / sigma_hat
    sizes["naive"] = np.sqrt(sizing.target_variance) / sigma_hat

    # 2. Conservative sizing: account for prediction error
    #    sigma_total = sigma_hat + z_alpha * RMSE
    z = norm.ppf(sizing.confidence_level)
    sigma_conservative = sigma_hat + z * model_rmse
    sizes["conservative"] = np.sqrt(sizing.target_variance) / sigma_conservative

    # 3. Blended sizing: shrink prediction toward historical mean
    if sizing.blending_factor is not None:
        sigma_blended = (
            sizing.blending_factor * sigma_hat
            + (1 - sizing.blending_factor) * sigma_train_mean
        )
        sizes["blended"] = np.sqrt(sizing.target_variance) / sigma_blended
    else:
        sizes["blended"] = sizes["naive"]

    # 4. Apply leverage caps
    for key in sizes:
        sizes[key] = np.clip(sizes[key], sizing.min_leverage, sizing.max_leverage)

    return sizes


def print_sizing_summary(
    sigma_hat: np.ndarray,
    sizes: dict,
    y_test: np.ndarray,
    sizing: SizingConfig,
) -> None:
    """Print a summary of position sizing results."""
    console.print()
    console.print(
        f"[bold cyan]Target Variance:[/bold cyan] {sizing.target_variance:.6f}"
    )
    console.print(
        f"[bold cyan]Target Vol (sqrt):[/bold cyan] {np.sqrt(sizing.target_variance):.4f}"
    )
    console.print(
        f"[bold cyan]Confidence Level:[/bold cyan] {sizing.confidence_level:.0%}"
    )
    console.print()

    # Summary statistics
    table = Table(title="Position Sizing Summary", box=box.ROUNDED)
    table.add_column("Strategy", style="cyan")
    table.add_column("Mean Size", style="green", justify="right")
    table.add_column("Std Size", style="green", justify="right")
    table.add_column("Min Size", style="green", justify="right")
    table.add_column("Max Size", style="green", justify="right")
    table.add_column("Realized Var", style="yellow", justify="right")
    table.add_column("Var Error", style="yellow", justify="right")

    for name, w in sizes.items():
        # Realized variance of sized positions: (w * sigma_realized)^2
        realized_var = np.mean((w * y_test) ** 2)
        var_error = realized_var - sizing.target_variance

        table.add_row(
            name,
            f"{np.mean(w):.2f}x",
            f"{np.std(w):.2f}x",
            f"{np.min(w):.2f}x",
            f"{np.max(w):.2f}x",
            f"{realized_var:.6f}",
            f"{var_error:+.6f}",
        )

    console.print(table)

    # Day-by-day example (first 10 test days)
    console.print()
    day_table = Table(
        title="First 10 Test Days — Position Sizing Detail", box=box.ROUNDED
    )
    day_table.add_column("Day", style="dim", justify="right")
    day_table.add_column("Predicted RV", style="cyan", justify="right")
    day_table.add_column("Actual RV", style="cyan", justify="right")
    day_table.add_column("Naive Size", style="green", justify="right")
    day_table.add_column("Conservative", style="green", justify="right")
    day_table.add_column("Blended", style="green", justify="right")
    day_table.add_column("Realized Var", style="yellow", justify="right")

    n_show = min(10, len(sigma_hat))
    for i in range(n_show):
        realized_var_naive = (sizes["naive"][i] * y_test[i]) ** 2
        day_table.add_row(
            str(i + 1),
            f"{sigma_hat[i]:.4f}",
            f"{y_test[i]:.4f}",
            f"{sizes['naive'][i]:.2f}x",
            f"{sizes['conservative'][i]:.2f}x",
            f"{sizes['blended'][i]:.2f}x",
            f"{realized_var_naive:.6f}",
        )

    console.print(day_table)


def main():
    config = tyro.cli(PipelineConfig)
    sizing = SizingConfig()

    print_header("Volatility-Targeted Position Sizing Example")
    print_info(f"Model: Lasso (R² ≈ 0.34 on test set)")
    print_info(f"Target variance: {sizing.target_variance:.6f}")

    # Run pipeline to get predictions
    pipeline = Pipeline(config)
    full_results, _, _ = pipeline.run()

    # Extract Lasso predictions
    lasso_results = full_results["lasso"]
    sigma_hat = lasso_results["predictions"]
    y_test = pipeline.y_test.values

    # Compute model RMSE on test set
    residuals = y_test - sigma_hat
    model_rmse = np.sqrt(np.mean(residuals**2))
    sigma_train_mean = pipeline.y_train.mean()

    print_section("Position Sizing Analysis")
    print_info(f"Model RMSE: {model_rmse:.4f}")
    print_info(f"Train mean RV: {sigma_train_mean:.4f}")
    print_info(f"Test mean RV: {y_test.mean():.4f}")

    # Compute position sizes
    sizes = compute_position_sizes(
        sigma_hat=sigma_hat,
        sigma_train_mean=sigma_train_mean,
        model_rmse=model_rmse,
        sizing=sizing,
    )

    # Print results
    print_sizing_summary(sigma_hat, sizes, y_test, sizing)

    print_success("Example complete")

    # Print usage tips
    print_section("How to Use in Production")
    console.print("""
[bold]Daily workflow:[/bold]
1. Run the pipeline before 9:30 AM to get today's sigma_hat
2. Choose a sizing strategy based on your risk tolerance:
   - [green]naive[/green]: Simple, but may overshoot target on high-vol days
   - [green]conservative[/green]: Accounts for prediction error, safer but lower returns
   - [green]blended[/green]: Shrinks toward historical mean, good middle ground
3. Size your position: position_notional = base_notional × leverage
4. Rebalance: Adjust position size as new predictions come in

[bold]Example:[/bold]
  Target variance = 0.0001 (1bp for the 30-min window)
  Predicted RV = 0.008
  Naive leverage = sqrt(0.0001) / 0.008 = 1.25x
  Conservative leverage = sqrt(0.0001) / (0.008 + 1.28 × 0.0035) = 0.80x

[bold]Key insight:[/bold]
  With R² = 0.34, your predictions explain 34% of variance.
  The remaining 66% is noise — expect realized variance to deviate
  from target by ~30-50% on individual days. Over many days,
  the average realized variance will converge toward the target.
""")


if __name__ == "__main__":
    main()
