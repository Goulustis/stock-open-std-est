from dataclasses import dataclass, field
from typing import Type
import numpy as np
import pandas as pd

from config import BaseModelConfig
from models.base import BasePredictor


@dataclass
class RollingMeanConfig(BaseModelConfig):
    """Config for RollingMean baseline."""

    _target: Type = field(default_factory=lambda: RollingMean)

    window: int = 10
    """Number of trailing days for rolling mean."""


class RollingMean(BasePredictor):
    """Predicts RV as the rolling mean of the last N days."""

    def __init__(self, config: RollingMeanConfig):
        super().__init__(config)
        self.config: RollingMeanConfig = config
        self.y_train = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.y_train = y.values
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        window = self.config.window
        rolling_mean = np.mean(self.y_train[-window:])

        return np.full(len(X), rolling_mean)

    def uses_features(self) -> bool:
        return False
