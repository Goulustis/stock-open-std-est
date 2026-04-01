from dataclasses import dataclass, field
from typing import Type, Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import tyro

from config import BaseConfig, BaseModelConfig
from data import load_and_filter
from features import build_feature_matrix
from models.base import BasePredictor
from models.naive import NaiveConfig, NaivePrevDay
from models.rolling_mean import RollingMeanConfig, RollingMean
from models.premarket import PremarketConfig, PremarketRV
from models.ewma import EwmaConfig, EwmaPredictor
from models.har_rv import HarRvConfig, HarRvPredictor
from models.garch import (
    Garch11Config,
    Garch11Predictor,
    EgarchConfig,
    EgarchPredictor,
)
from models.lasso import LassoConfig, LassoPredictor
from models.ridge import RidgeConfig, RidgePredictor
from models.random_forest import RandomForestConfig, RandomForestPredictor
from models.lightgbm import LightGBMConfig, LightGBMPredictor
from models.xgboost import XGBoostConfig, XGBoostPredictor
from utils import (
    console,
    print_header,
    print_section,
    print_success,
    print_info,
    print_warning,
    print_data_summary,
)


@tyro.conf.configure(tyro.conf.SuppressFixed)
@dataclass
class PipelineConfig(BaseConfig):
    """Pipeline configuration with all model sub-configs."""

    _target: Optional[Type] = field(default=None, repr=False)

    naive_config: NaiveConfig = field(default_factory=NaiveConfig)
    rolling_config: RollingMeanConfig = field(default_factory=RollingMeanConfig)
    premarket_config: PremarketConfig = field(default_factory=PremarketConfig)
    ewma_config: EwmaConfig = field(default_factory=EwmaConfig)
    har_config: HarRvConfig = field(default_factory=HarRvConfig)
    garch_config: Garch11Config = field(default_factory=Garch11Config)
    egarch_config: EgarchConfig = field(default_factory=EgarchConfig)
    lasso_config: LassoConfig = field(default_factory=LassoConfig)
    ridge_config: RidgeConfig = field(default_factory=RidgeConfig)
    rf_config: RandomForestConfig = field(default_factory=RandomForestConfig)
    lgbm_config: LightGBMConfig = field(default_factory=LightGBMConfig)
    xgb_config: XGBoostConfig = field(default_factory=XGBoostConfig)

    config_f: Optional[str] = None
    """Path to YAML config file to load and override defaults."""

    def __post_init__(self):
        if self.config_f is not None:
            from config import load_config_from_yaml

            loaded = load_config_from_yaml(self.config_f)
            if not isinstance(loaded, dict):
                raise TypeError(f"config_f {self.config_f} did not load into a dict")
            self.__dict__.update(loaded)


class Pipeline:
    """Orchestrates data loading, feature building, model training, and prediction."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.predictors: Dict[str, BasePredictor] = {}
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None

        self.populate_modules()

    def populate_modules(self):
        """Load data, build features, split train/test, instantiate predictors."""
        print_section("Loading Data")
        df = load_and_filter(self.config)
        print_info(f"Loaded {len(df)} bars, {df['date_et'].nunique()} trading days")

        print_section("Building Features")
        feature_df, target = build_feature_matrix(df, self.config)
        print_data_summary(feature_df, "Feature Matrix")
        print_info(f"Target: {len(target)} values, mean={target.mean():.4f}, std={target.std():.4f}")

        print_section("Train/Test Split")
        self.X_train, self.X_test, self.y_train, self.y_test = self._time_split(
            feature_df, target, self.config.train_end_date
        )
        print_info(f"Train: {len(self.X_train)} samples ({self.X_train.index.min()} to {self.X_train.index.max()})")
        print_info(f"Test:  {len(self.X_test)} samples ({self.X_test.index.min()} to {self.X_test.index.max()})")

        self.feature_names = list(self.X_train.columns)

        print_section("Standardizing Features")
        self.scaler = StandardScaler()
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            index=self.X_train.index,
            columns=self.X_train.columns,
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            index=self.X_test.index,
            columns=self.X_test.columns,
        )
        print_success("Features standardized (fit on train, transform on test)")

        print_section("Instantiating Predictors")
        self.predictors = {
            "naive": self.config.naive_config.setup(),
            "rolling": self.config.rolling_config.setup(),
            "premarket": self.config.premarket_config.setup(),
            "ewma": self.config.ewma_config.setup(),
            "har_rv": self.config.har_config.setup(),
            "garch": self.config.garch_config.setup(),
            "egarch": self.config.egarch_config.setup(),
            "lasso": self.config.lasso_config.setup(),
            "ridge": self.config.ridge_config.setup(),
            "rf": self.config.rf_config.setup(),
            "lgbm": self.config.lgbm_config.setup(),
            "xgb": self.config.xgb_config.setup(),
        }
        for name, pred in self.predictors.items():
            uses_feat = pred.uses_features()
            print_info(f"  {name}: {pred.get_name()} (uses features: {uses_feat})")

    def _time_split(self, feature_df: pd.DataFrame, target: pd.Series, train_end_date: str) -> tuple:
        """Split data by date. All rows <= train_end_date go to train."""
        train_dates = feature_df.index <= pd.to_datetime(train_end_date).date()
        test_dates = ~train_dates

        X_train = feature_df[train_dates].copy()
        X_test = feature_df[test_dates].copy()
        y_train = target[train_dates].copy()
        y_test = target[test_dates].copy()

        aligned = X_train.index.intersection(y_train.index)
        X_train = X_train.loc[aligned]
        y_train = y_train.loc[aligned]

        aligned = X_test.index.intersection(y_test.index)
        X_test = X_test.loc[aligned]
        y_test = y_test.loc[aligned]

        valid_train = ~y_train.isna()
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]

        valid_test = ~y_test.isna()
        X_test = X_test[valid_test]
        y_test = y_test[valid_test]

        return X_train, X_test, y_train, y_test

    def _train_predict(
        self,
        predictors: Dict[str, BasePredictor],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        feature_set_name: str = "features",
    ) -> Dict[str, Dict[str, Any]]:
        """Train a set of predictors and generate predictions."""
        print_section(f"Training Models ({feature_set_name})")

        results = {}
        for name, predictor in predictors.items():
            print_info(f"Training {name}...")

            X_tr = X_train if predictor.uses_features() else X_train
            X_te = X_test if predictor.uses_features() else X_test

            predictor.fit(X_tr, y_train)
            predictions = predictor.predict(X_te)

            results[name] = {
                "predictions": predictions,
                "predictor": predictor,
            }
            print_success(f"  {name}: done")

        return results

    def run(
        self,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], List[str]]:
        """Run full pipeline: full features, then Lasso-selected features.

        Returns:
            full_results: results from training on all features
            selected_results: results from training on Lasso-selected features
            selected_features: list of feature names selected by Lasso
        """
        full_results = self._train_predict(
            self.predictors,
            self.X_train_scaled,
            self.X_test_scaled,
            self.y_train,
            self.y_test,
            "Full Features",
        )

        # Use Adaptive Lasso for feature selection (oracle property)
        lasso_pred = self.predictors["lasso"]
        selected_features = getattr(lasso_pred, "selected_features_", self.feature_names)

        print_section(f"Lasso Feature Selection: {len(selected_features)}/{len(self.feature_names)} features selected")
        for i, feat in enumerate(selected_features, 1):
            print_info(f"  {i}. {feat}")

        X_train_sel = self.X_train_scaled[selected_features]
        X_test_sel = self.X_test_scaled[selected_features]

        selected_predictors = {}
        skipped = []
        for name, pred in self.predictors.items():
            if not pred.uses_features():
                selected_predictors[name] = pred
                continue
            required = getattr(pred, "pm_feature_col", None)
            if required and required not in selected_features:
                skipped.append(name)
                continue
            selected_predictors[name] = pred

        if skipped:
            print_warning(f"Skipped in selected-features pass (required features not selected): {skipped}")

        selected_results = self._train_predict(
            selected_predictors,
            X_train_sel,
            X_test_sel,
            self.y_train,
            self.y_test,
            f"Selected Features ({len(selected_features)})",
        )

        return full_results, selected_results, selected_features
