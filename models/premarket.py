from dataclasses import dataclass, field
from typing import Type
import numpy as np
import pandas as pd

from config import BaseModelConfig
from models.base import BasePredictor


@dataclass
class PremarketConfig(BaseModelConfig):
    """Config for PremarketRV baseline — predicts from same-day premarket vol."""

    _target: Type = field(default_factory=lambda: PremarketRV)

    pm_feature_col: str = "pm_rv"
    """Column name for premarket RV in the feature matrix."""


class PremarketRV(BasePredictor):
    """Predicts 9:30-10:00 RV using same-day premarket realized volatility.

    This is a strong baseline because premarket vol is highly correlated
    with opening vol.
    """

    def __init__(self, config: PremarketConfig):
        super().__init__(config)
        self.config: PremarketConfig = config
        self.pm_feature_col = config.pm_feature_col
        self.intercept = 0.0
        self.slope = 1.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit a simple linear regression: target = slope * pm_rv + intercept."""
        if self.pm_feature_col not in X.columns:
            raise ValueError(f"Feature '{self.pm_feature_col}' not found in X")

        pm_rv = X[self.pm_feature_col].values
        mask = ~np.isnan(pm_rv) & ~np.isnan(y.values)

        if mask.sum() < 10:
            self.slope = 1.0
            self.intercept = y.mean()
        else:
            x = pm_rv[mask]
            yt = y.values[mask]

            x_mean = x.mean()
            y_mean = yt.mean()

            numerator = np.sum((x - x_mean) * (yt - y_mean))
            denominator = np.sum((x - x_mean) ** 2)

            if denominator > 0:
                self.slope = numerator / denominator
                self.intercept = y_mean - self.slope * x_mean
            else:
                self.slope = 1.0
                self.intercept = 0.0

        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        pm_rv = X[self.pm_feature_col].values
        predictions = self.slope * pm_rv + self.intercept

        return predictions

    def uses_features(self) -> bool:
        return True
