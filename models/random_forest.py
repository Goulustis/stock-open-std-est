from dataclasses import dataclass, field
from typing import Type
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from config import BaseModelConfig
from models.base import BasePredictor


@dataclass
class RandomForestConfig(BaseModelConfig):
    """Config for Random Forest regression model."""

    _target: Type = field(default_factory=lambda: RandomForestPredictor)

    n_estimators: int = 200
    """Number of trees."""

    max_depth: int = 10
    """Maximum tree depth."""

    random_state: int = 42


class RandomForestPredictor(BasePredictor):
    """Random Forest regression model."""

    def __init__(self, config: RandomForestConfig):
        super().__init__(config)
        self.config: RandomForestConfig = config
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        mask = ~y.isna() & ~X.isna().any(axis=1)
        X_clean = X[mask].values
        y_clean = y[mask].values

        self.model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_clean, y_clean)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X.values)

    def uses_features(self) -> bool:
        return True
