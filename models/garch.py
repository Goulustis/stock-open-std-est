from dataclasses import dataclass, field
from typing import Type, Optional
import numpy as np
import pandas as pd

from config import BaseModelConfig
from models.base import BasePredictor


@dataclass
class Garch11Config(BaseModelConfig):
    """Config for GARCH(1,1) volatility prediction model."""

    _target: Optional[Type] = field(default_factory=lambda: Garch11Predictor)


class Garch11Predictor(BasePredictor):
    """GARCH(1,1) model for volatility prediction.

    Models variance as:
        sigma2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma2_{t-1}

    Uses the arch package for estimation.
    """

    def __init__(self, config: Garch11Config):
        super().__init__(config)
        self.config: Garch11Config = config
        self.fitted_model = None
        self._params = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        from arch import arch_model

        model = arch_model(y.values, vol="GARCH", p=1, q=1)
        result = model.fit(disp="off")
        self.fitted_model = model
        self._params = result.params
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        n = len(X)
        forecast = self.fitted_model.forecast(params=self._params, horizon=n)
        variance_forecast = forecast.variance.iloc[-1].values
        return np.sqrt(variance_forecast)

    def uses_features(self) -> bool:
        return False

    def get_name(self) -> str:
        return "GARCH(1,1)"


@dataclass
class EgarchConfig(BaseModelConfig):
    """Config for EGARCH volatility prediction model."""

    _target: Optional[Type] = field(default_factory=lambda: EgarchPredictor)


class EgarchPredictor(BasePredictor):
    """EGARCH model for volatility prediction.

    Models log variance as:
        log(sigma2_t) = omega + alpha * |epsilon_{t-1}|/sigma_{t-1}
                        + gamma * (epsilon_{t-1}/sigma_{t-1})
                        + beta * log(sigma2_{t-1})

    The gamma parameter captures asymmetry (leverage effect).
    Uses the arch package for estimation.
    """

    def __init__(self, config: EgarchConfig):
        super().__init__(config)
        self.config: EgarchConfig = config
        self.fitted_model = None
        self._params = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        from arch import arch_model

        model = arch_model(y.values, vol="EGARCH", p=1, q=1)
        result = model.fit(disp="off")
        self.fitted_model = model
        self._params = result.params
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        n = len(X)
        forecast = self.fitted_model.forecast(
            params=self._params, horizon=n, method="simulation"
        )
        variance_forecast = forecast.variance.iloc[-1].values
        return np.sqrt(variance_forecast)

    def uses_features(self) -> bool:
        return False

    def get_name(self) -> str:
        return "EGARCH"
