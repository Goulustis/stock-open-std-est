from dataclasses import dataclass, field
from typing import Type, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
from sklearn.linear_model import LinearRegression

from config import BaseModelConfig
from models.base import BasePredictor


@dataclass
class LassoConfig(BaseModelConfig):
    """Config for Lasso (L1-regularized) regression model."""

    _target: Optional[Type] = field(default_factory=lambda: LassoPredictor)

    n_alphas: int = 100
    """Number of alpha values to try in CV."""

    cv_folds: int = 5
    """Number of cross-validation folds."""


class LassoPredictor(BasePredictor):
    """Lasso regression with automatic feature selection via LassoCV.

    After fitting, features with non-zero coefficients are considered
    "selected" and stored in self.selected_features_.
    """

    def __init__(self, config: LassoConfig):
        super().__init__(config)
        self.config: LassoConfig = config
        self.model = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.feature_names_ = list(X.columns)

        mask = ~y.isna() & ~X.isna().any(axis=1)
        X_clean = X[mask].values
        y_clean = y[mask].values

        alphas = np.logspace(-6, -1, self.config.n_alphas)

        self.model = LassoCV(
            alphas=alphas,
            cv=self.config.cv_folds,
            max_iter=10000,
            random_state=42,
        )
        self.model.fit(X_clean, y_clean)

        self.selected_features_ = [
            name
            for name, coef in zip(self.feature_names_, self.model.coef_)
            if abs(coef) > 1e-10
        ]

        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        return self.model.predict(X.values)

    def uses_features(self) -> bool:
        return True


@dataclass
class AdaptiveLassoConfig(BaseModelConfig):
    """Config for Adaptive Lasso (Audrino & Knaus, 2016).

    Two-step procedure:
    1. Fit OLS to get initial coefficient estimates
    2. Fit Lasso with penalty weights w_j = 1 / |beta_OLS_j|^gamma

    This gives the oracle property — consistent variable selection.
    """

    _target: Optional[Type] = field(default_factory=lambda: AdaptiveLassoPredictor)

    n_alphas: int = 100
    """Number of alpha values to try in CV."""

    cv_folds: int = 5
    """Number of cross-validation folds."""

    gamma: float = 1.0
    """Power for adaptive weights. gamma=1 is standard."""


class AdaptiveLassoPredictor(BasePredictor):
    """Adaptive Lasso regression (Audrino & Knaus, 2016).

    Two-step procedure:
    1. Fit OLS to get initial coefficient estimates
    2. Fit weighted Lasso with penalty w_j = 1 / |beta_OLS_j|^gamma

    This gives the oracle property — consistent variable selection.
    Features with zero coefficients are considered "unselected."
    """

    def __init__(self, config: AdaptiveLassoConfig):
        super().__init__(config)
        self.config: AdaptiveLassoConfig = config
        self.model = None
        self.feature_names_ = None
        self.ols_model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.feature_names_ = list(X.columns)

        mask = ~y.isna() & ~X.isna().any(axis=1)
        X_clean = X[mask].values
        y_clean = y[mask].values

        # Step 1: OLS for initial estimates
        self.ols_model = LinearRegression()
        self.ols_model.fit(X_clean, y_clean)
        beta_ols = self.ols_model.coef_

        # Step 2: Compute adaptive weights w_j = 1 / |beta_OLS_j|^gamma
        # Clip to avoid division by zero
        abs_beta = np.abs(beta_ols)
        min_beta = (
            np.percentile(abs_beta[abs_beta > 0], 10) if np.any(abs_beta > 0) else 1e-10
        )
        abs_beta = np.clip(abs_beta, min_beta, None)
        adaptive_weights = 1.0 / (abs_beta**self.config.gamma)

        # Step 3: Weighted Lasso via feature scaling
        # Lasso penalizes |beta_j|, so scaling X_j by w_j makes the penalty w_j * |beta_j|
        X_weighted = X_clean * adaptive_weights[np.newaxis, :]

        alphas = np.logspace(-6, -1, self.config.n_alphas)

        self.model = LassoCV(
            alphas=alphas,
            cv=self.config.cv_folds,
            max_iter=10000,
            random_state=42,
        )
        self.model.fit(X_weighted, y_clean)

        # Recover original coefficients: beta_original = beta_weighted * w_j
        self.coef_ = self.model.coef_ * adaptive_weights
        self.intercept_ = self.model.intercept_

        self.selected_features_ = [
            name
            for name, coef in zip(self.feature_names_, self.coef_)
            if abs(coef) > 1e-10
        ]

        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        return X.values @ self.coef_ + self.intercept_

    def uses_features(self) -> bool:
        return True
