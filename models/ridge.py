from dataclasses import dataclass, field
from typing import Type
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from config import BaseModelConfig
from models.base import BasePredictor


@dataclass
class RidgeConfig(BaseModelConfig):
    """Config for Ridge (L2-regularized) regression model."""

    _target: Type = field(default_factory=lambda: RidgePredictor)

    alphas: tuple = field(default_factory=lambda: (0.001, 0.01, 0.1, 1.0, 10.0, 100.0))
    """Alpha values to try in CV."""


class RidgePredictor(BasePredictor):
    """Ridge regression with automatic alpha selection via RidgeCV."""

    def __init__(self, config: RidgeConfig):
        super().__init__(config)
        self.config: RidgeConfig = config
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        mask = ~y.isna() & ~X.isna().any(axis=1)
        X_clean = X[mask].values
        y_clean = y[mask].values

        self.model = RidgeCV(alphas=self.config.alphas)
        self.model.fit(X_clean, y_clean)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X.values)

    def uses_features(self) -> bool:
        return True
