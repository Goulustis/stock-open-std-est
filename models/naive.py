from dataclasses import dataclass, field
from typing import Type
import numpy as np
import pandas as pd

from config import BaseModelConfig
from models.base import BasePredictor


@dataclass
class NaiveConfig(BaseModelConfig):
    """Config for NaivePrevDay baseline — predicts yesterday's RV."""

    _target: Type = field(default_factory=lambda: NaivePrevDay)


class NaivePrevDay(BasePredictor):
    """Predicts that today's RV equals yesterday's RV.

    This is the strongest baseline for volatility prediction due to
    volatility clustering. Hard to beat.
    """

    def __init__(self, config: NaiveConfig):
        super().__init__(config)
        self.y_train = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Store training target for persistence prediction."""
        self.y_train = y.values
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict each day's RV as the previous day's actual RV.

        For the first test day, uses the last training day's RV.
        For subsequent test days, uses the previous test day's actual RV
        (which we don't have at prediction time). So we use a rolling approach:
        predict yesterday's prediction for day 2+, or use last train RV throughout.

        In practice, the standard approach is: pred_t = y_{t-1} for all t.
        Since we don't know y_{t-1} for test days beyond the first,
        we use the last known y (from training) for all predictions.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        return np.full(len(X), self.y_train[-1])

    def uses_features(self) -> bool:
        return False
