from dataclasses import dataclass, field
from typing import Type, Optional
import numpy as np
import pandas as pd

from config import BaseModelConfig
from models.base import BasePredictor


@dataclass
class EwmaConfig(BaseModelConfig):
    """Config for EWMA volatility prediction model."""

    _target: Optional[Type] = field(default_factory=lambda: EwmaPredictor)

    lambda_param: float = 0.94
    """Decay factor for EWMA (typically 0.94 for daily data)."""


class EwmaPredictor(BasePredictor):
    """Exponentially Weighted Moving Average volatility model.

    Models variance as:
        sigma2_t = lambda * sigma2_{t-1} + (1 - lambda) * r^2_{t-1}

    Predicts volatility (sqrt of variance) for the next day.
    """

    def __init__(self, config: EwmaConfig):
        super().__init__(config)
        self.config: EwmaConfig = config
        self.y_train = None
        self.last_ewma_var = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Store training data and compute EWMA state through training period."""
        self.y_train = y.values
        lam = self.config.lambda_param

        # Initialize variance as mean of squared training RVs
        sigma2 = np.mean(self.y_train**2)

        # Run EWMA recursion through training data
        for r in self.y_train:
            sigma2 = lam * sigma2 + (1 - lam) * r**2

        self.last_ewma_var = sigma2
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate EWMA volatility predictions for test data.

        Continues the EWMA recursion from the final training state.
        Returns sqrt of predicted variances (volatility, not variance).
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        lam = self.config.lambda_param
        n = len(X)
        predictions = np.zeros(n)

        sigma2 = self.last_ewma_var
        for i in range(n):
            # Predict next day variance from current state
            sigma2 = lam * sigma2 + (1 - lam) * self.y_train[-1] ** 2
            predictions[i] = np.sqrt(sigma2)

        return predictions

    def uses_features(self) -> bool:
        return False
