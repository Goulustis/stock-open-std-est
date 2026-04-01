from dataclasses import dataclass, field
from typing import Type, Optional
import numpy as np
import pandas as pd
import xgboost as xgb

from config import BaseModelConfig
from models.base import BasePredictor


@dataclass
class XGBoostConfig(BaseModelConfig):
    """Config for XGBoost regression model."""

    _target: Optional[Type] = field(default_factory=lambda: XGBoostPredictor)

    n_estimators: int = 200
    learning_rate: float = 0.05
    max_depth: int = 6
    random_state: int = 42


class XGBoostPredictor(BasePredictor):
    """XGBoost regression model."""

    def __init__(self, config: XGBoostConfig):
        super().__init__(config)
        self.config: XGBoostConfig = config
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        mask = ~y.isna() & ~X.isna().any(axis=1)
        X_clean = X[mask].values
        y_clean = y[mask].values

        self.model = xgb.XGBRegressor(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
            verbosity=0,
        )
        self.model.fit(X_clean, y_clean)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X.values)

    def uses_features(self) -> bool:
        return True
