from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, mean_squared_error, r2_score

from .config import AppConfig
from .features import FEATURE_COLUMNS, build_feature_table, training_rows


MODEL_FILE = "proxy_model.joblib"
METRICS_FILE = "metrics.json"
DEFAULT_HORIZONS = [3, 5, 10, 20, 60, 120]


class WeightedEnsembleRegressor:
    def __init__(self, models: list[Any], weights: list[float], target_clip: tuple[float, float] | None = None) -> None:
        self.models = models
        total = float(sum(max(0.0, weight) for weight in weights))
        if total <= 0:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = [max(0.0, weight) / total for weight in weights]
        self.target_clip = target_clip

    def predict(self, rows):  # noqa: ANN001
        preds = np.column_stack([model.predict(rows) for model in self.models])
        out = np.average(preds, axis=1, weights=np.asarray(self.weights, dtype=float))
        if self.target_clip is not None:
            out = np.clip(out, self.target_clip[0], self.target_clip[1])
        return out


class ConstantProbabilityClassifier:
    def __init__(self, probability: float) -> None:
        self.probability = float(np.clip(probability, 0.0, 1.0))
        self.classes_ = np.asarray([0, 1])

    def predict_proba(self, rows):  # noqa: ANN001
        p = np.full(len(rows), self.probability, dtype=float)
        return np.column_stack([1.0 - p, p])

    def predict(self, rows):  # noqa: ANN001
        return (self.predict_proba(rows)[:, 1] >= 0.5).astype(int)


class WeightedEnsembleClassifier:
    def __init__(self, models: list[Any], weights: list[float]) -> None:
        self.models = models
        total = float(sum(max(0.0, weight) for weight in weights))
        if total <= 0:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = [max(0.0, weight) / total for weight in weights]
        self.classes_ = np.asarray([0, 1])

    def predict_proba(self, rows):  # noqa: ANN001
        probs = np.column_stack([_positive_probability_from_model(model, rows) for model in self.models])
        p = np.average(probs, axis=1, weights=np.asarray(self.weights, dtype=float))
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def predict(self, rows):  # noqa: ANN001
        return (self.predict_proba(rows)[:, 1] >= 0.5).astype(int)


def _positive_probability_from_model(model: Any, rows) -> np.ndarray:  # noqa: ANN001
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(rows), dtype=float)
        if proba.ndim == 1:
            return np.clip(proba, 0.0, 1.0)
        classes = list(getattr(model, "classes_", [0, 1]))
        positive_index = classes.index(1) if 1 in classes else proba.shape[1] - 1
        return np.clip(proba[:, positive_index], 0.0, 1.0)
    pred = np.asarray(model.predict(rows), dtype=float)
    return np.clip(pred, 0.0, 1.0)


def predict_positive_probability(model: Any, rows) -> np.ndarray:  # noqa: ANN001
    return _positive_probability_from_model(model, rows)


def _lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401

        return True
    except ImportError:
        return False


def _make_regressors(random_state: int, kind: str) -> tuple[list[Any], str]:
    try:
        from lightgbm import LGBMRegressor  # type: ignore

        if kind == "drawdown":
            models = [
                LGBMRegressor(
                    objective="quantile",
                    alpha=0.25,
                    n_estimators=650,
                    learning_rate=0.026,
                    num_leaves=31,
                    min_child_samples=45,
                    subsample=0.88,
                    colsample_bytree=0.86,
                    reg_alpha=0.08,
                    reg_lambda=1.25,
                    random_state=random_state,
                    n_jobs=1,
                    verbosity=-1,
                    force_col_wise=True,
                ),
                LGBMRegressor(
                    objective="regression_l1",
                    n_estimators=520,
                    learning_rate=0.03,
                    num_leaves=23,
                    min_child_samples=65,
                    subsample=0.82,
                    colsample_bytree=0.78,
                    reg_alpha=0.16,
                    reg_lambda=1.8,
                    random_state=random_state + 17,
                    n_jobs=1,
                    verbosity=-1,
                    force_col_wise=True,
                ),
                LGBMRegressor(
                    objective="regression",
                    n_estimators=620,
                    learning_rate=0.026,
                    num_leaves=31,
                    min_child_samples=55,
                    subsample=0.86,
                    colsample_bytree=0.82,
                    reg_alpha=0.10,
                    reg_lambda=1.40,
                    random_state=random_state + 29,
                    n_jobs=1,
                    verbosity=-1,
                    force_col_wise=True,
                ),
            ]
            return models, "advanced_lgbm_weighted_quantile_ensemble"
        return (
            [
                LGBMRegressor(
                    objective="huber",
                    alpha=0.86,
                    n_estimators=720,
                    learning_rate=0.024,
                    num_leaves=31,
                    min_child_samples=45,
                    subsample=0.88,
                    colsample_bytree=0.86,
                    reg_alpha=0.08,
                    reg_lambda=1.20,
                    random_state=random_state,
                    n_jobs=1,
                    verbosity=-1,
                    force_col_wise=True,
                ),
                LGBMRegressor(
                    objective="regression_l1",
                    n_estimators=560,
                    learning_rate=0.030,
                    num_leaves=23,
                    min_child_samples=65,
                    subsample=0.82,
                    colsample_bytree=0.78,
                    reg_alpha=0.18,
                    reg_lambda=1.70,
                    random_state=random_state + 13,
                    n_jobs=1,
                    verbosity=-1,
                    force_col_wise=True,
                ),
                LGBMRegressor(
                    objective="regression",
                    n_estimators=760,
                    learning_rate=0.025,
                    num_leaves=31,
                    min_child_samples=50,
                    subsample=0.86,
                    colsample_bytree=0.84,
                    reg_alpha=0.10,
                    reg_lambda=1.10,
                    random_state=random_state + 29,
                    n_jobs=1,
                    verbosity=-1,
                    force_col_wise=True,
                ),
            ],
            "advanced_lgbm_weighted_ensemble",
        )
    except ImportError:
        return (
            [
                HistGradientBoostingRegressor(
                    loss="absolute_error",
                    max_iter=450,
                    learning_rate=0.04,
                    l2_regularization=0.06,
                    max_leaf_nodes=31,
                    random_state=random_state,
                ),
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    max_iter=380,
                    learning_rate=0.035,
                    l2_regularization=0.12,
                    max_leaf_nodes=23,
                    random_state=random_state + 13,
                ),
            ],
            "sklearn_hist_gradient_weighted_ensemble",
        )


def _make_classifiers(random_state: int, kind: str) -> tuple[list[Any], str]:
    try:
        from lightgbm import LGBMClassifier  # type: ignore

        if kind == "trend":
            return (
                [
                    LGBMClassifier(
                        objective="binary",
                        n_estimators=620,
                        learning_rate=0.026,
                        num_leaves=31,
                        min_child_samples=55,
                        subsample=0.86,
                        colsample_bytree=0.84,
                        reg_alpha=0.12,
                        reg_lambda=1.45,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=1,
                        verbosity=-1,
                        force_col_wise=True,
                    ),
                    LGBMClassifier(
                        objective="binary",
                        n_estimators=520,
                        learning_rate=0.032,
                        num_leaves=23,
                        min_child_samples=75,
                        subsample=0.82,
                        colsample_bytree=0.78,
                        reg_alpha=0.20,
                        reg_lambda=1.90,
                        class_weight="balanced",
                        random_state=random_state + 19,
                        n_jobs=1,
                        verbosity=-1,
                        force_col_wise=True,
                    ),
                ],
                "advanced_lgbm_trend_classifier_ensemble",
            )
        return (
            [
                LGBMClassifier(
                    objective="binary",
                    n_estimators=640,
                    learning_rate=0.025,
                    num_leaves=31,
                    min_child_samples=50,
                    subsample=0.88,
                    colsample_bytree=0.86,
                    reg_alpha=0.10,
                    reg_lambda=1.35,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=1,
                    verbosity=-1,
                    force_col_wise=True,
                ),
                LGBMClassifier(
                    objective="binary",
                    n_estimators=540,
                    learning_rate=0.030,
                    num_leaves=23,
                    min_child_samples=70,
                    subsample=0.82,
                    colsample_bytree=0.78,
                    reg_alpha=0.18,
                    reg_lambda=1.80,
                    class_weight="balanced",
                    random_state=random_state + 13,
                    n_jobs=1,
                    verbosity=-1,
                    force_col_wise=True,
                ),
            ],
            "advanced_lgbm_direction_classifier_ensemble",
        )
    except ImportError:
        return (
            [
                HistGradientBoostingClassifier(
                    loss="log_loss",
                    max_iter=360,
                    learning_rate=0.04,
                    l2_regularization=0.08,
                    max_leaf_nodes=31,
                    random_state=random_state,
                ),
                HistGradientBoostingClassifier(
                    loss="log_loss",
                    max_iter=300,
                    learning_rate=0.035,
                    l2_regularization=0.14,
                    max_leaf_nodes=23,
                    random_state=random_state + 13,
                ),
            ],
            "sklearn_hist_gradient_classifier_ensemble",
        )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _probability_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, p)),
        "positive_rate": float(np.mean(y)),
        "predicted_positive_rate": float(np.mean(pred)),
    }


def _confidence_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    target_accuracy: float,
    min_coverage: float,
) -> dict[str, float | bool | int]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    pred = (p >= 0.5).astype(int)
    confidence = np.abs(p - 0.5) * 2.0
    overall_accuracy = float(accuracy_score(y, pred))

    candidates = sorted(
        {
            *[float(x) for x in np.linspace(0.0, 0.95, 20)],
            *[float(x) for x in np.quantile(confidence, np.linspace(0.0, 0.95, 20))],
        }
    )
    target_choice: dict[str, float | bool | int] | None = None
    best_choice: dict[str, float | bool | int] | None = None
    for threshold in candidates:
        mask = confidence >= threshold
        count = int(mask.sum())
        if count <= 0:
            continue
        coverage = float(mask.mean())
        if coverage < min_coverage:
            continue
        accuracy = float(accuracy_score(y[mask], pred[mask]))
        item: dict[str, float | bool | int] = {
            "target_accuracy": float(target_accuracy),
            "selected_confidence_threshold": float(threshold),
            "high_confidence_accuracy": accuracy,
            "high_confidence_coverage": coverage,
            "high_confidence_count": count,
            "target_met": bool(accuracy >= target_accuracy),
            "overall_accuracy": overall_accuracy,
        }
        if accuracy >= target_accuracy:
            if target_choice is None or coverage > float(target_choice["high_confidence_coverage"]):
                target_choice = item
        if best_choice is None:
            best_choice = item
        else:
            best_acc = float(best_choice["high_confidence_accuracy"])
            best_cov = float(best_choice["high_confidence_coverage"])
            if accuracy > best_acc or (accuracy == best_acc and coverage > best_cov):
                best_choice = item

    choice = target_choice or best_choice
    if choice is None:
        choice = {
            "target_accuracy": float(target_accuracy),
            "selected_confidence_threshold": 1.0,
            "high_confidence_accuracy": overall_accuracy,
            "high_confidence_coverage": 1.0,
            "high_confidence_count": int(len(y)),
            "target_met": bool(overall_accuracy >= target_accuracy),
            "overall_accuracy": overall_accuracy,
        }
    return choice


def _clip_bounds(y: pd.Series, lower: float = 0.01, upper: float = 0.99) -> tuple[float, float]:
    low = float(y.quantile(lower))
    high = float(y.quantile(upper))
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        return float(y.min()), float(y.max())
    return low, high


def _clip_target(y: pd.Series, bounds: tuple[float, float]) -> pd.Series:
    return y.clip(lower=bounds[0], upper=bounds[1])


def _sample_weights(frame: pd.DataFrame, horizon: int) -> np.ndarray:
    dates = pd.to_datetime(frame["date"])
    age_days = (dates.max() - dates).dt.days.clip(lower=0)
    half_life = max(180, min(900, horizon * 45))
    recency = np.exp(-np.log(2) * age_days / half_life)
    volatility = pd.to_numeric(frame.get("volatility_20", 0.0), errors="coerce").fillna(0.0).clip(lower=0.003, upper=0.18)
    calm_weight = 1.0 / np.sqrt(1.0 + volatility.to_numpy() * 20.0)
    weights = 0.35 + 0.65 * recency.to_numpy()
    weights = weights * calm_weight
    return weights / np.mean(weights)


def _fit_weighted_ensemble(
    model_specs: list[Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    sample_weight: np.ndarray,
    target_bounds: tuple[float, float],
) -> tuple[WeightedEnsembleRegressor, np.ndarray, list[dict[str, float]]]:
    clipped_train = _clip_target(y_train, target_bounds)
    fitted: list[Any] = []
    preds: list[np.ndarray] = []
    model_scores: list[dict[str, float]] = []
    weights: list[float] = []

    for spec in model_specs:
        model = clone(spec)
        model.fit(x_train, clipped_train, sample_weight=sample_weight)
        pred = np.clip(model.predict(x_valid), target_bounds[0], target_bounds[1])
        mae = float(mean_absolute_error(y_valid, pred))
        rmse = float(mean_squared_error(y_valid, pred) ** 0.5)
        direction_hit = float(np.mean((pred > 0) == (y_valid.to_numpy() > 0)))
        fitted.append(model)
        preds.append(pred)
        model_scores.append({"mae": mae, "rmse": rmse, "direction_hit_rate": direction_hit})
        direction_bonus = max(0.35, min(1.35, 0.45 + direction_hit))
        weights.append(direction_bonus / max(mae, 1e-6))

    ensemble = WeightedEnsembleRegressor(fitted, weights, target_clip=target_bounds)
    pred_matrix = np.column_stack(preds)
    ensemble_pred = np.average(pred_matrix, axis=1, weights=np.asarray(ensemble.weights, dtype=float))
    ensemble_pred = np.clip(ensemble_pred, target_bounds[0], target_bounds[1])
    return ensemble, ensemble_pred, model_scores


def _fit_final_ensemble(
    model_specs: list[Any],
    weights: list[float],
    x_all: pd.DataFrame,
    y_all: pd.Series,
    sample_weight: np.ndarray,
    target_bounds: tuple[float, float],
) -> WeightedEnsembleRegressor:
    clipped = _clip_target(y_all, target_bounds)
    fitted: list[Any] = []
    for spec in model_specs:
        model = clone(spec)
        model.fit(x_all, clipped, sample_weight=sample_weight)
        fitted.append(model)
    return WeightedEnsembleRegressor(fitted, weights, target_clip=target_bounds)


def _fit_weighted_classifier(
    model_specs: list[Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    sample_weight: np.ndarray,
) -> tuple[WeightedEnsembleClassifier, np.ndarray, list[dict[str, float]], str]:
    y_train_int = y_train.astype(int)
    y_valid_int = y_valid.astype(int)
    if y_train_int.nunique() < 2:
        constant = ConstantProbabilityClassifier(float(y_train_int.mean()))
        pred = predict_positive_probability(constant, x_valid)
        return WeightedEnsembleClassifier([constant], [1.0]), pred, [_probability_metrics(y_valid_int.to_numpy(), pred)], "constant"

    fitted: list[Any] = []
    preds: list[np.ndarray] = []
    model_scores: list[dict[str, float]] = []
    weights: list[float] = []
    for spec in model_specs:
        model = clone(spec)
        model.fit(x_train, y_train_int, sample_weight=sample_weight)
        pred = predict_positive_probability(model, x_valid)
        scores = _probability_metrics(y_valid_int.to_numpy(), pred)
        fitted.append(model)
        preds.append(pred)
        model_scores.append(scores)
        weights.append(max(0.05, scores["accuracy"]) / max(scores["brier"], 1e-4))

    ensemble = WeightedEnsembleClassifier(fitted, weights)
    ensemble_pred = predict_positive_probability(ensemble, x_valid)
    ensemble_accuracy = float(accuracy_score(y_valid_int.to_numpy(), (ensemble_pred >= 0.5).astype(int)))
    prior_probability = float(y_train_int.mean())
    prior_pred = np.full(len(x_valid), prior_probability, dtype=float)
    prior_accuracy = float(accuracy_score(y_valid_int.to_numpy(), (prior_pred >= 0.5).astype(int)))
    if prior_accuracy > ensemble_accuracy:
        prior = ConstantProbabilityClassifier(prior_probability)
        prior_scores = _probability_metrics(y_valid_int.to_numpy(), prior_pred)
        prior_scores["fallback_from_accuracy"] = ensemble_accuracy
        return WeightedEnsembleClassifier([prior], [1.0]), prior_pred, [prior_scores], "prior_fallback"
    return ensemble, ensemble_pred, model_scores, "ensemble"


def _fit_final_classifier(
    model_specs: list[Any],
    weights: list[float],
    x_all: pd.DataFrame,
    y_all: pd.Series,
    sample_weight: np.ndarray,
    strategy: str,
) -> WeightedEnsembleClassifier:
    y_all_int = y_all.astype(int)
    if y_all_int.nunique() < 2 or strategy in {"constant", "prior_fallback"}:
        return WeightedEnsembleClassifier([ConstantProbabilityClassifier(float(y_all_int.mean()))], [1.0])
    fitted: list[Any] = []
    for spec in model_specs:
        model = clone(spec)
        model.fit(x_all, y_all_int, sample_weight=sample_weight)
        fitted.append(model)
    return WeightedEnsembleClassifier(fitted, weights)


def _date_split(df: pd.DataFrame, validation_fraction: float) -> tuple[pd.Series, pd.Series]:
    dates = np.array(sorted(df["date"].dropna().unique()))
    if len(dates) < 20:
        raise ValueError("Not enough unique dates for a time-based validation split.")
    split_index = max(1, int(len(dates) * (1 - validation_fraction)))
    split_index = min(split_index, len(dates) - 1)
    cutoff = dates[split_index]
    train_mask = df["date"] < cutoff
    valid_mask = df["date"] >= cutoff
    return train_mask, valid_mask


def _training_horizons(cfg: AppConfig) -> list[int]:
    return sorted({int(h) for h in [*DEFAULT_HORIZONS, cfg.model.horizon_days] if int(h) > 0})


def _train_horizon(history: pd.DataFrame, cfg: AppConfig, horizon: int, seed_offset: int) -> tuple[dict[str, Any], dict[str, Any]]:
    feature_table = build_feature_table(history, horizon=horizon)
    trainable = training_rows(feature_table, horizon=horizon)

    counts = trainable.groupby("symbol").size()
    keep_symbols = counts[counts >= cfg.model.min_rows_per_symbol].index
    trainable = trainable[trainable["symbol"].isin(keep_symbols)].copy()
    if trainable.empty:
        raise ValueError("No symbol has enough rows after feature engineering.")

    train_mask, valid_mask = _date_split(trainable, cfg.model.validation_fraction)
    train_df = trainable.loc[train_mask].copy()
    valid_df = trainable.loc[valid_mask].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError("Train/validation split produced an empty partition.")

    x_train = train_df[FEATURE_COLUMNS]
    x_valid = valid_df[FEATURE_COLUMNS]
    y_ret_train = train_df[f"target_return_{horizon}d"]
    y_ret_valid = valid_df[f"target_return_{horizon}d"]
    y_dir_train = (y_ret_train > 0).astype(int)
    y_dir_valid = (y_ret_valid > 0).astype(int)
    y_trend_train = (train_df[f"target_trend_return_{horizon}d"] > 0).astype(int)
    y_trend_valid = (valid_df[f"target_trend_return_{horizon}d"] > 0).astype(int)
    y_dd_train = train_df[f"target_drawdown_{horizon}d"]
    y_dd_valid = valid_df[f"target_drawdown_{horizon}d"]

    ret_bounds = _clip_bounds(y_ret_train, 0.01, 0.99)
    dd_bounds = _clip_bounds(y_dd_train, 0.01, 0.99)
    train_weight = _sample_weights(train_df, horizon)

    ret_specs, backend = _make_regressors(cfg.model.random_state + seed_offset, "return")
    dd_specs, _ = _make_regressors(cfg.model.random_state + seed_offset + 7, "drawdown")
    direction_specs, direction_backend = _make_classifiers(cfg.model.random_state + seed_offset + 13, "direction")
    trend_specs, trend_backend = _make_classifiers(cfg.model.random_state + seed_offset + 19, "trend")
    ret_validation_model, pred_ret, ret_model_scores = _fit_weighted_ensemble(
        ret_specs,
        x_train,
        y_ret_train,
        x_valid,
        y_ret_valid,
        train_weight,
        ret_bounds,
    )
    dd_validation_model, pred_dd, dd_model_scores = _fit_weighted_ensemble(
        dd_specs,
        x_train,
        y_dd_train,
        x_valid,
        y_dd_valid,
        train_weight,
        dd_bounds,
    )
    direction_validation_model, pred_direction_prob, direction_model_scores, direction_strategy = _fit_weighted_classifier(
        direction_specs,
        x_train,
        y_dir_train,
        x_valid,
        y_dir_valid,
        train_weight,
    )
    trend_validation_model, pred_trend_prob, trend_model_scores, trend_strategy = _fit_weighted_classifier(
        trend_specs,
        x_train,
        y_trend_train,
        x_valid,
        y_trend_valid,
        train_weight,
    )
    direction_hit = np.mean((pred_ret > 0) == (y_ret_valid.to_numpy() > 0))
    direction_confidence = _confidence_metrics(
        y_dir_valid.to_numpy(),
        pred_direction_prob,
        target_accuracy=0.65,
        min_coverage=0.15,
    )
    trend_confidence = _confidence_metrics(
        y_trend_valid.to_numpy(),
        pred_trend_prob,
        target_accuracy=0.75,
        min_coverage=0.10,
    )

    validation = valid_df[
        [
            "date",
            "symbol",
            "close",
            f"target_return_{horizon}d",
            f"target_trend_return_{horizon}d",
            f"target_drawdown_{horizon}d",
        ]
    ].copy()
    validation["pred_return"] = pred_ret
    validation["pred_drawdown"] = pred_dd
    validation["pred_direction_prob"] = pred_direction_prob
    validation["pred_trend_prob"] = pred_trend_prob
    validation["score"] = (
        validation["pred_return"] / validation["pred_drawdown"].abs().clip(lower=0.01)
        + (validation["pred_direction_prob"] - 0.5) * 1.50
        + (validation["pred_trend_prob"] - 0.5) * 1.00
    )
    top_daily = (
        validation.sort_values(["date", "score"], ascending=[True, False])
        .groupby("date")
        .head(min(cfg.risk.max_portfolio_positions, 5))
    )

    metrics = {
        "backend": backend,
        "direction_backend": direction_backend,
        "trend_backend": trend_backend,
        "horizon_days": horizon,
        "training_strategy": "time_decay_winsorized_regression_direction_trend_classifier_refit_all_data",
        "symbols": sorted(trainable["symbol"].unique().tolist()),
        "feature_count": len(FEATURE_COLUMNS),
        "rows": {
            "train": int(len(train_df)),
            "validation": int(len(valid_df)),
        },
        "date_range": {
            "train_start": str(train_df["date"].min().date()),
            "train_end": str(train_df["date"].max().date()),
            "validation_start": str(valid_df["date"].min().date()),
            "validation_end": str(valid_df["date"].max().date()),
        },
        "return_model": _metrics(y_ret_valid.to_numpy(), pred_ret),
        "drawdown_model": _metrics(y_dd_valid.to_numpy(), pred_dd),
        "direction_classifier": _probability_metrics(y_dir_valid.to_numpy(), pred_direction_prob),
        "trend_classifier": _probability_metrics(y_trend_valid.to_numpy(), pred_trend_prob),
        "return_target_clip": {"low": ret_bounds[0], "high": ret_bounds[1]},
        "drawdown_target_clip": {"low": dd_bounds[0], "high": dd_bounds[1]},
        "return_base_models": ret_model_scores,
        "drawdown_base_models": dd_model_scores,
        "direction_base_models": direction_model_scores,
        "trend_base_models": trend_model_scores,
        "direction_classifier_strategy": direction_strategy,
        "trend_classifier_strategy": trend_strategy,
        "return_ensemble_weights": ret_validation_model.weights,
        "drawdown_ensemble_weights": dd_validation_model.weights,
        "direction_ensemble_weights": direction_validation_model.weights,
        "trend_ensemble_weights": trend_validation_model.weights,
        "direction_hit_rate": float(direction_hit),
        "direction_confidence": direction_confidence,
        "trend_confidence": trend_confidence,
        "validation_top_daily_mean_forward_return": float(top_daily[f"target_return_{horizon}d"].mean()),
        "validation_top_daily_win_rate": float((top_daily[f"target_return_{horizon}d"] > 0).mean()),
    }
    all_weight = _sample_weights(trainable, horizon)
    x_all = trainable[FEATURE_COLUMNS]
    final_ret_model = _fit_final_ensemble(ret_specs, ret_validation_model.weights, x_all, trainable[f"target_return_{horizon}d"], all_weight, ret_bounds)
    final_dd_model = _fit_final_ensemble(dd_specs, dd_validation_model.weights, x_all, trainable[f"target_drawdown_{horizon}d"], all_weight, dd_bounds)
    final_direction_model = _fit_final_classifier(
        direction_specs,
        direction_validation_model.weights,
        x_all,
        (trainable[f"target_return_{horizon}d"] > 0).astype(int),
        all_weight,
        direction_strategy,
    )
    final_trend_model = _fit_final_classifier(
        trend_specs,
        trend_validation_model.weights,
        x_all,
        (trainable[f"target_trend_return_{horizon}d"] > 0).astype(int),
        all_weight,
        trend_strategy,
    )
    model_payload = {
        "return_model": final_ret_model,
        "drawdown_model": final_dd_model,
        "direction_model": final_direction_model,
        "trend_model": final_trend_model,
        "validation_return_model": ret_validation_model,
        "validation_drawdown_model": dd_validation_model,
        "validation_direction_model": direction_validation_model,
        "validation_trend_model": trend_validation_model,
        "confidence_thresholds": {
            "direction": direction_confidence,
            "trend": trend_confidence,
        },
        "horizon_days": horizon,
        "backend": backend,
        "direction_backend": direction_backend,
        "trend_backend": trend_backend,
        "direction_classifier_strategy": direction_strategy,
        "trend_classifier_strategy": trend_strategy,
    }
    return model_payload, metrics


def train_proxy_model(history: pd.DataFrame, cfg: AppConfig) -> dict[str, Any]:
    horizon_models: dict[str, dict[str, Any]] = {}
    horizon_metrics: dict[str, dict[str, Any]] = {}
    backend = ""
    for index, horizon in enumerate(_training_horizons(cfg)):
        model_payload, metrics_payload = _train_horizon(history, cfg, horizon, seed_offset=index * 31)
        horizon_models[str(horizon)] = model_payload
        horizon_metrics[str(horizon)] = metrics_payload
        backend = str(metrics_payload["backend"])

    default_horizon = int(cfg.model.horizon_days)
    if str(default_horizon) not in horizon_models:
        default_horizon = min((int(h) for h in horizon_models), key=lambda h: abs(h - cfg.model.horizon_days))
    default_model = horizon_models[str(default_horizon)]

    cfg.model.model_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "return_model": default_model["return_model"],
        "drawdown_model": default_model["drawdown_model"],
        "direction_model": default_model["direction_model"],
        "trend_model": default_model["trend_model"],
        "confidence_thresholds": default_model.get("confidence_thresholds", {}),
        "feature_columns": FEATURE_COLUMNS,
        "horizon_days": default_horizon,
        "available_horizons": sorted(int(h) for h in horizon_models),
        "horizon_models": horizon_models,
        "backend": backend,
        "model_type": "multi_horizon_directional_ensemble_proxy",
        "config": {
            "data": asdict(cfg.data),
            "model": asdict(cfg.model),
            "risk": asdict(cfg.risk),
        },
    }
    model_path = cfg.model.model_dir / MODEL_FILE
    metrics_path = cfg.model.model_dir / METRICS_FILE
    metrics = {
        "model_type": "multi_horizon_directional_ensemble_proxy",
        "backend": backend,
        "default_horizon_days": default_horizon,
        "available_horizons": sorted(int(h) for h in horizon_models),
        "horizons": horizon_metrics,
    }
    if str(default_horizon) in horizon_metrics:
        metrics.update(horizon_metrics[str(default_horizon)])
    joblib.dump(bundle, model_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "model_path": model_path,
        "metrics_path": metrics_path,
        "metrics": metrics,
    }


def load_model(model_dir: Path) -> dict[str, Any]:
    path = model_dir / MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)
