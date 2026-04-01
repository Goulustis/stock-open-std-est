from abc import ABC, abstractmethod
from typing import Optional, List
import numpy as np
import pandas as pd

from config import BaseModelConfig


class BasePredictor(ABC):
    """Base interface for all prediction models.

    All predictors must implement fit() and predict().
    Some models (baselines) don't use features and will ignore X.
    """

    def __init__(self, config: BaseModelConfig):
        self.config = config
        self.is_fitted = False
        self.selected_features_: Optional[List[str]] = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model on features X and target y."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for features X."""
        pass

    def get_name(self) -> str:
        """Return human-readable model name."""
        return self.__class__.__name__

    def uses_features(self) -> bool:
        """Whether this model uses feature data (vs being univariate/baseline)."""
        return True
