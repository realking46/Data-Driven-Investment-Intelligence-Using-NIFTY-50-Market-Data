from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EXCLUDED_FEATURE_COLUMNS = {
    "date",
    "symbol",
    "series",
    "source_file",
    "future_close_1d",
    "future_close_5d",
    "future_close_21d",
    "future_return_1d",
    "future_return_5d",
    "future_return_21d",
    "target_direction_1d",
    "target_direction_5d",
    "target_direction_21d",
}


@dataclass
class RidgeModel:
    feature_columns: list[str]
    means: pd.Series
    stds: pd.Series
    coefficients: np.ndarray
    intercept: float
    alpha: float
    target_column: str


def select_feature_columns(df: pd.DataFrame, target_column: str) -> list[str]:
    excluded = set(EXCLUDED_FEATURE_COLUMNS)
    excluded.add(target_column)

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [
        col
        for col in numeric_columns
        if col not in excluded
        and not col.startswith("future_")
        and not col.startswith("target_")
    ]
    return features


def prepare_model_frame(df: pd.DataFrame, target_column: str, min_non_null_ratio: float = 0.65) -> pd.DataFrame:
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")

    features = select_feature_columns(df, target_column)
    if not features:
        raise ValueError("No numeric feature columns are available for modeling.")

    out = df[["date", "symbol", target_column, *features]].copy()
    out = out.dropna(subset=["date", "symbol", target_column])

    usable_features = []
    for col in features:
        non_null_ratio = out[col].notna().mean()
        if non_null_ratio >= min_non_null_ratio:
            usable_features.append(col)

    if not usable_features:
        raise ValueError("No feature columns have enough non-null values for modeling.")

    return out[["date", "symbol", target_column, *usable_features]].sort_values(["date", "symbol"]).reset_index(drop=True)


def time_based_split(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()

    unique_dates = pd.Series(pd.to_datetime(df["date"]).sort_values().unique())
    split_index = int(len(unique_dates) * (1 - test_fraction))
    split_index = max(1, min(split_index, len(unique_dates) - 1))
    split_date = unique_dates.iloc[split_index]

    train = df[df["date"] < split_date].copy()
    test = df[df["date"] >= split_date].copy()
    return train, test


def _standardize(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    means = train_x.mean(numeric_only=True)
    stds = train_x.std(numeric_only=True).replace(0, 1).fillna(1)

    train_filled = train_x.fillna(means).fillna(0)
    test_filled = test_x.fillna(means).fillna(0)

    x_train = ((train_filled - means) / stds).to_numpy(dtype=float)
    x_test = ((test_filled - means) / stds).to_numpy(dtype=float)
    return x_train, x_test, means, stds


def fit_ridge_regression(train: pd.DataFrame, target_column: str, alpha: float = 1.0) -> RidgeModel:
    feature_columns = select_feature_columns(train, target_column)
    train_x = train[feature_columns]
    train_y = train[target_column].to_numpy(dtype=float)

    x_train, _, means, stds = _standardize(train_x, train_x)
    design = np.column_stack([np.ones(len(x_train)), x_train])

    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0
    weights = np.linalg.pinv(design.T @ design + penalty) @ design.T @ train_y

    return RidgeModel(
        feature_columns=feature_columns,
        means=means,
        stds=stds,
        coefficients=weights[1:],
        intercept=float(weights[0]),
        alpha=alpha,
        target_column=target_column,
    )


def predict(model: RidgeModel, frame: pd.DataFrame) -> np.ndarray:
    x = frame[model.feature_columns].fillna(model.means).fillna(0)
    x_scaled = ((x - model.means) / model.stds).to_numpy(dtype=float)
    return model.intercept + (x_scaled @ model.coefficients)


def evaluate_predictions(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> dict[str, float]:
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    error = y_true - y_pred

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    denominator = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - (np.sum(error**2) / denominator)) if denominator != 0 else np.nan
    directional_accuracy = float(np.mean(np.sign(y_true) == np.sign(y_pred)))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "directional_accuracy": directional_accuracy,
    }


def baseline_mean_prediction(train: pd.DataFrame, test: pd.DataFrame, target_column: str) -> np.ndarray:
    symbol_means = train.groupby("symbol")[target_column].mean()
    global_mean = train[target_column].mean()
    return test["symbol"].map(symbol_means).fillna(global_mean).to_numpy(dtype=float)


def _sample_training_frame(train: pd.DataFrame, max_rows: int = 70_000) -> pd.DataFrame:
    if len(train) <= max_rows:
        return train
    return train.sort_values("date").tail(max_rows).reset_index(drop=True)


def _imputed_xy(train: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str], target_column: str):
    train_x = train[feature_columns]
    test_x = test[feature_columns]
    means = train_x.mean(numeric_only=True)
    train_x = train_x.fillna(means).fillna(0)
    test_x = test_x.fillna(means).fillna(0)
    train_y = train[target_column].to_numpy(dtype=float)
    return train_x, test_x, train_y


def sklearn_model_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    import os

    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

    try:
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    except ModuleNotFoundError:
        return {}, pd.DataFrame()

    sampled_train = _sample_training_frame(train)
    train_x, test_x, train_y = _imputed_xy(sampled_train, test, feature_columns, target_column)

    models = {
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=140,
            learning_rate=0.055,
            max_leaf_nodes=24,
            l2_regularization=0.05,
            random_state=42,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=60,
            max_depth=8,
            min_samples_leaf=100,
            max_features=0.65,
            n_jobs=1,
            random_state=42,
        ),
    }

    predictions: dict[str, np.ndarray] = {}
    importance_frames: list[pd.DataFrame] = []

    for model_name, model in models.items():
        model.fit(train_x, train_y)
        predictions[model_name] = model.predict(test_x)

        if hasattr(model, "feature_importances_"):
            importance_frames.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "feature": feature_columns,
                        "importance": model.feature_importances_,
                    }
                )
            )

    if importance_frames:
        importance = pd.concat(importance_frames, ignore_index=True)
        importance = importance.sort_values(["model", "importance"], ascending=[True, False]).reset_index(drop=True)
    else:
        importance = pd.DataFrame()

    return predictions, importance


def coefficient_importance(model: RidgeModel, top_n: int = 25) -> pd.DataFrame:
    importance = pd.DataFrame(
        {
            "model": "ridge_regression",
            "feature": model.feature_columns,
            "coefficient": model.coefficients,
            "absolute_coefficient": np.abs(model.coefficients),
            "importance": np.abs(model.coefficients),
        }
    )
    return importance.sort_values("absolute_coefficient", ascending=False).head(top_n).reset_index(drop=True)


def train_and_evaluate(
    features: pd.DataFrame,
    target_column: str = "future_return_5d",
    alpha: float = 5.0,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_frame = prepare_model_frame(features, target_column)
    train, test = time_based_split(model_frame, test_fraction=test_fraction)

    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty dataset. Add more history or adjust the split.")

    baseline_pred = baseline_mean_prediction(train, test, target_column)
    ridge_model = fit_ridge_regression(train, target_column=target_column, alpha=alpha)
    ridge_pred = predict(ridge_model, test)
    feature_columns = ridge_model.feature_columns
    sklearn_predictions, sklearn_importance = sklearn_model_predictions(train, test, target_column, feature_columns)

    prediction_map = {
        "symbol_mean_baseline": baseline_pred,
        "ridge_regression": ridge_pred,
        **sklearn_predictions,
    }

    metrics = []
    for model_name, prediction in prediction_map.items():
        row = {
            "model": model_name,
            "target": target_column,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_start": train["date"].min(),
            "train_end": train["date"].max(),
            "test_start": test["date"].min(),
            "test_end": test["date"].max(),
        }
        row.update(evaluate_predictions(test[target_column], prediction))
        metrics.append(row)

    metrics_df = pd.DataFrame(metrics).sort_values("rmse", ascending=True).reset_index(drop=True)
    best_model = metrics_df.iloc[0]["model"]

    predictions = test[["date", "symbol", target_column]].copy()
    for model_name, prediction in prediction_map.items():
        predictions[f"{model_name}_prediction"] = prediction
    predictions["best_model"] = best_model
    predictions["best_model_prediction"] = prediction_map[best_model]
    predictions["actual_direction"] = np.where(predictions[target_column] > 0, 1, 0)
    predictions["best_model_direction"] = np.where(predictions["best_model_prediction"] > 0, 1, 0)

    importance = coefficient_importance(ridge_model)
    if not sklearn_importance.empty:
        importance = pd.concat([importance, sklearn_importance], ignore_index=True, sort=False)

    return metrics_df, predictions, importance
