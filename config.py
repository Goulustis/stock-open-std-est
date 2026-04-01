from dataclasses import dataclass, field, fields, MISSING
from typing import Type, Optional, Tuple, Any
import tyro
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()


class PrintableConfig:
    """Base config class with rich-formatted string representation."""

    def __str__(self):
        lines = [f"[bold]{self.__class__.__name__}:[/bold]"]
        for key, val in vars(self).items():
            if key.startswith("_"):
                continue
            if isinstance(val, Tuple):
                val = ", ".join(str(v) for v in val)
            elif isinstance(val, PrintableConfig):
                val = val.__class__.__name__
            lines.append(f"  [cyan]{key}:[/cyan] [green]{val}[/green]")
        return "\n".join(lines)


@dataclass
class BaseModelConfig(PrintableConfig):
    """Base config for all prediction models."""

    _target: Optional[Type] = field(default=None, repr=False)

    def setup(self, **kwargs) -> Any:
        """Instantiate the predictor class from config."""
        return self._target(self, **kwargs)

    def get_name(self) -> str:
        return self.__class__.__name__.replace("Config", "")


@dataclass
class BaseConfig(PrintableConfig):
    """Top-level configuration for the std_pred pipeline."""

    _target: Optional[Type] = field(default=None, repr=False)

    # Data paths
    csv_path: str = "NVDA_5m.csv"
    """Path to the 5-minute bar CSV file."""

    # Time windows (ET)
    premarket_start: str = "08:00"
    """Start of premarket window for features."""

    premarket_end: str = "09:25"
    """End of premarket window for features."""

    target_start: str = "09:30"
    """Start of target window (9:30-10:00 RV)."""

    target_end: str = "10:00"
    """End of target window (9:30-10:00 RV)."""

    # Train/test split
    train_end_date: str = "2023-12-31"
    """Last date included in training set (inclusive)."""

    # Feature computation
    rv_windows: Tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    """Lag windows for RV features (in trading days)."""

    rolling_mean_windows: Tuple[int, ...] = (5, 10, 20)
    """Windows for rolling mean RV features."""

    # Model sub-configs (populated by PipelineConfig)
    # These are placeholders; PipelineConfig overrides with actual model configs

    def validate(self) -> None:
        """Validate configuration consistency."""
        assert Path(self.csv_path).exists(), f"CSV file not found: {self.csv_path}"
        assert self.premarket_start < self.target_start, "Premarket must end before target starts"
        assert self.target_start < self.target_end, "Target start must be before target end"


def load_config_from_yaml(filename: str, base_config: BaseConfig = None) -> dict:
    """Load config overrides from a YAML file."""
    import yaml

    config_dict = yaml.load(Path(filename).read_text(), Loader=yaml.Loader)
    return config_dict


def create_cli_config() -> "BaseConfig":
    """Create configuration from CLI arguments using tyro."""
    return tyro.cli(BaseConfig)
