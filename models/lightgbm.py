from dataclasses import dataclass, field
from typing import Type
import numpy as np
import pandas as pd
import lightgbm as lgb

from config import BaseModelConfig
from models.base import BasePredictor


@dataclass
class LightGBMConfig(BaseModelConfig):
    """Config for LightGBM regression model."""

    _target: Type = field(default_factory=lambda: LightGBMPredictor)

    n_estimators: int = 200
    learning_rate: float = 0.05
    max_depth: int = 6
    num_leaves: int = 31
    random_state: int = 42


class LightGBMPredictor(BasePredictor):
    """LightGBM regression model."""

    def __init__(self, config: LightGBMConfig):
        super().__init__(config)
        self.config: LightGBMConfig = config
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        mask = ~y.isna() & ~X.isna().any(axis=1)
        X_clean = X[mask].values
        y_clean = y[mask].values

        self.model = lgb.LGBMRegressor(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            num_leaves=self.config.num_leaves,
            random_state=self.config.random_state,
            verbose=-1,
        )
        self.model.fit(X_clean, y_clean, feature_name=list(X[mask].columns))
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X.values)

    def uses_features(self) -> bool:
        return True
