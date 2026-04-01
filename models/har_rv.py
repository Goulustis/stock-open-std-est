from dataclasses import dataclass, field
from typing import Type, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import BaseModelConfig
from models.base import BasePredictor


@dataclass
class HarRvConfig(BaseModelConfig):
    """Config for HAR-RV-X (Heterogeneous Autoregressive Realized Volatility with eXogenous features)."""

    _target: Optional[Type] = field(default_factory=lambda: HarRvPredictor)

    exogenous_features: List[str] = field(
        default_factory=lambda: ["pm_rv", "overnight_gap_abs"]
    )
    """Exogenous features to include beyond HAR lags."""

    use_exogenous: bool = True
    """Whether to include exogenous features."""


class HarRvPredictor(BasePredictor):
    """HAR-RV-X model for volatility prediction.

    Combines daily, weekly (5-day), and monthly (22-day) lagged RV components
    with optional exogenous features in an OLS regression framework.
    """

    def __init__(self, config: HarRvConfig):
        super().__init__(config)
        self.config: HarRvConfig = config
        self.model = None
        self.y_train = None
        self.feature_names_: List[str] = []

    def _build_har_features(self, y_values: np.ndarray) -> pd.DataFrame:
        """Build HAR daily/weekly/monthly features from a sequence of RV values."""
        y_series = pd.Series(y_values)
        features = pd.DataFrame(index=range(len(y_values)))
        features["rv_daily"] = y_series.shift(1).values
        features["rv_weekly"] = y_series.shift(1).rolling(5).mean().values
        features["rv_monthly"] = y_series.shift(1).rolling(22).mean().values
        return features

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.y_train = y.values
        har_features = self._build_har_features(self.y_train)
        har_features.index = y.index

        if self.config.use_exogenous and self.config.exogenous_features:
            avail = [f for f in self.config.exogenous_features if f in X.columns]
            X_exog = X[avail].copy()
            feature_matrix = pd.concat([har_features, X_exog], axis=1)
            self.feature_names_ = list(har_features.columns) + avail
        else:
            feature_matrix = har_features
            self.feature_names_ = list(har_features.columns)

        mask = ~feature_matrix.isna().any(axis=1)
        self.model = LinearRegression()
        self.model.fit(feature_matrix[mask].values, y.values[mask.values])
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        n = len(X)
        full_y = np.concatenate([self.y_train, np.zeros(n)])
        predictions = np.zeros(n)

        for i in range(n):
            features = self._build_har_features(full_y[: len(self.y_train) + i + 1])
            last_features = features.iloc[[-1]].values

            if self.config.use_exogenous and self.config.exogenous_features:
                avail = [f for f in self.config.exogenous_features if f in X.columns]
                exog = X[avail].iloc[[i]].values
                last_features = np.hstack([last_features, exog])

            pred = self.model.predict(last_features)[0]
            predictions[i] = max(pred, 0)
            full_y[len(self.y_train) + i] = pred

        return predictions

    def get_name(self) -> str:
        return "HAR-RV-X"
