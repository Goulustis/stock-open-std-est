from typing import Dict, Any, List
import numpy as np
import pandas as pd
from pathlib import Path

from utils import (
    console,
    print_header,
    print_section,
    print_success,
    print_info,
    print_metrics_table,
    print_comparison_table,
    print_feature_importance,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics.

    Returns:
        dict with R2, MAE, RMSE, MAPE
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {"R2": np.nan, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}

    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)

    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    mape = np.mean(np.abs(residuals / y_true)) * 100 if np.all(y_true > 0) else np.nan

    return {
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
    }


def evaluate_all_results(
    full_results: Dict[str, Dict[str, Any]],
    selected_results: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    selected_features: List[str],
    feature_names: List[str],
    output_dir: str = ".",
) -> tuple:
    """Evaluate both full and selected feature results.

    Returns:
        (full_metrics, selected_metrics) dicts
    """
    print_section("Evaluation Results")

    full_metrics = {}
    for name, result in full_results.items():
        preds = result["predictions"]
        full_metrics[name] = compute_metrics(y_test, preds)

    selected_metrics = {}
    for name, result in selected_results.items():
        preds = result["predictions"]
        selected_metrics[name] = compute_metrics(y_test, preds)

    print_comparison_table(full_metrics, selected_metrics)

    save_results_csv(full_metrics, selected_metrics, output_dir)

    lasso_pred = full_results["lasso"]["predictor"]
    if hasattr(lasso_pred, "model") and lasso_pred.model is not None:
        coef_dict = dict(zip(lasso_pred.feature_names_, lasso_pred.model.coef_))
        print_feature_importance(
            coef_dict, "Lasso Coefficients (Feature Selection)", top_n=20
        )
        print_feature_importance(coef_dict, "Adaptive Lasso Coefficients", top_n=20)

    lasso_pred = full_results["lasso"]["predictor"]
    if hasattr(lasso_pred, "model") and lasso_pred.model is not None:
        coef_dict = dict(zip(lasso_pred.feature_names_, lasso_pred.model.coef_))
        print_feature_importance(coef_dict, "Lasso Coefficients", top_n=20)

    rf_pred = full_results["rf"]["predictor"]
    if hasattr(rf_pred, "model") and rf_pred.model is not None:
        rf_importance = dict(zip(feature_names, rf_pred.model.feature_importances_))
        print_feature_importance(
            rf_importance, "Random Forest Feature Importance", top_n=20
        )

    lgbm_pred = full_results["lgbm"]["predictor"]
    if hasattr(lgbm_pred, "model") and lgbm_pred.model is not None:
        lgbm_importance = dict(zip(feature_names, lgbm_pred.model.feature_importances_))
        print_feature_importance(
            lgbm_importance, "LightGBM Feature Importance", top_n=20
        )

    return full_metrics, selected_metrics


def save_results_csv(
    full_metrics: Dict[str, Dict[str, float]],
    selected_metrics: Dict[str, Dict[str, float]],
    output_dir: str = ".",
) -> Path:
    """Save metrics to CSV file."""
    full_df = pd.DataFrame(full_metrics).T
    full_df["feature_set"] = "full"
    selected_df = pd.DataFrame(selected_metrics).T
    selected_df["feature_set"] = "selected"

    combined = pd.concat([full_df, selected_df])
    output_path = Path(output_dir) / "results.csv"
    combined.to_csv(output_path)
    print_success(f"Results saved to {output_path}")
    return output_path
