"""
Общие функции для старых ноутбуков проекта.

Цель модуля — не переписывать ноутбуки в новый пайплайн, а убрать из них
повторяющийся код: генерацию cd-файлов CatBoost, метрики, сезонные признаки,
разбиение по месяцам, обучение моделей и сохранение importance.
"""

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd

from catboost import Pool, CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# Коэффициенты сезонности POS. Вынесены из train_pos/train_pos_binary/train_pos_delta.
POS_VALUES = {
    "2025-01-01": 18.64458142,
    "2025-02-01": 16.77100784,
    "2025-03-01": 17.91131324,
    "2025-04-01": 16.83359701,
    "2025-05-01": 17.84970137,
    "2025-06-01": 15.84694422,
    "2025-07-01": 16.14898008,
    "2025-08-01": 17.55541702,
    "2025-09-01": 17.42844075,
    "2025-10-01": 18.64065285,
    "2025-11-01": 18.40382364,
    "2025-12-01": 17.37507281,
    "2026-01-01": 16.99701678,
    "2026-02-01": 18.5416244,
    "2026-03-01": 17.56242237,
    "2026-04-01": 16.51183347,
    "2026-05-01": 17.08078586,
}

# Базовые POS-признаки, для которых строятся сезонно скорректированные версии.
SEASON_BASE_COLS = [
    "total_spend_lag_1m",
    "total_spend_lag_2m",
    "total_spend_lag_3m",
    "total_spend_lag_4m",
    "total_spend_lag_5m",
    "total_spend_lag_6m",
    "total_spend_3m",
    "total_spend_6m",
    "pos_turnover_lag_1m",
    "pos_turnover_3m",
    "pos_turnover_6m",
]


def ensure_project_imports():
    """Заглушка для явного вызова в ноутбуках: импорт уже настроен в ячейке."""
    return True


def auto_cd(df, target_col, weight_col=None, ignore_cols=None, output_path="cd.txt"):
    """
    Генерирует column_description-файл для CatBoost.

    target_col — колонка таргета;
    weight_col — колонка веса, если используется;
    ignore_cols — служебные колонки, которые CatBoost должен игнорировать.
    """
    ignore_cols = set(ignore_cols or [])
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    with open(output_path, "w", encoding="utf-8") as f:
        for i, col in enumerate(df.columns):
            if col == target_col:
                col_type = "Label"
            elif weight_col is not None and col == weight_col:
                col_type = "Weight"
            elif col in ignore_cols:
                col_type = "Auxiliary"
            elif col in cat_cols:
                col_type = "Categ"
            else:
                col_type = "Num"
            f.write(f"{i}\t{col_type}\n")


def regression_metrics(y_true, y_pred, prefix=""):
    """Считает основные метрики регрессии."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "r2": r2_score(y_true, y_pred),
        "median_ae": median_absolute_error(y_true, y_pred),
    }
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics


def classification_metrics(y_true, p_pred, threshold=0.5, prefix=""):
    """Считает основные метрики бинарной классификации по вероятностям."""
    y_true = np.asarray(y_true).reshape(-1)
    p_pred = np.asarray(p_pred).reshape(-1)
    y_hat = (p_pred >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_true, p_pred),
        "pr_auc": average_precision_score(y_true, p_pred),
        "logloss": log_loss(y_true, p_pred),
        "brier": brier_score_loss(y_true, p_pred),
        "precision@0.5": precision_score(y_true, y_hat, zero_division=0),
        "recall@0.5": recall_score(y_true, y_hat, zero_division=0),
        "f1@0.5": f1_score(y_true, y_hat, zero_division=0),
        "pred_positive_rate": y_hat.mean(),
        "true_positive_rate": y_true.mean(),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
    metrics.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics


def get_honest_season_coef(forecast_month, pos_values=None, missing_value=1.0):
    """POS(m-12) / POS(m-14). Never uses POS from forecast year.

    Missing historical pairs return neutral 1.0 (not current/future POS).
    Inspect historical POS coverage before training; absent pairs return 1.0.
    """
    values = POS_VALUES if pos_values is None else pos_values
    series = pd.Series(values, dtype="float64")
    series.index = pd.to_datetime(series.index).to_period("M").to_timestamp()
    m = pd.Timestamp(forecast_month).to_period("M").to_timestamp()
    numerator = series.get(m - pd.DateOffset(months=12), np.nan)
    denominator = series.get(m - pd.DateOffset(months=14), np.nan)
    if pd.notna(numerator) and pd.notna(denominator) and denominator > 0:
        return float(numerator / denominator)
    return float(missing_value)


def add_standard_time_features(df, month_col="forecast_month", season_base_cols=None,
                               pos_values=None, clip_min=0.75, clip_max=1.50):
    """The same leakage-free seasonality for training and inference.

    Historical POS series must be the SAME for both methods. A missing
    m-12/m-14 pair yields neutral 1.0 (no forecast-year POS fallback).
    """
    result = df.copy()
    result[month_col] = pd.to_datetime(result[month_col])
    values = POS_VALUES if pos_values is None else pos_values
    series = pd.Series(values, dtype="float64")
    series.index = pd.to_datetime(series.index).to_period("M").to_timestamp()
    months = result[month_col].dt.to_period("M").dt.to_timestamp()
    previous_year = months - pd.DateOffset(months=12)
    previous_year_base = months - pd.DateOffset(months=14)
    numerator = previous_year.map(series)
    denominator = previous_year_base.map(series)
    available = numerator.notna() & denominator.notna() & (denominator > 0)
    coef = (numerator / denominator).where(available, 1.0)
    result["season_total_pos_coef_honest"] = coef.clip(clip_min, clip_max).astype("float64")

    base_cols = list(SEASON_BASE_COLS if season_base_cols is None else season_base_cols)
    for col in base_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").astype("float64")
            result[f"{col}_season_adj"] = result[col].fillna(0) * result["season_total_pos_coef_honest"]

    result["forecast_month_num"] = result[month_col].dt.month
    result["forecast_quarter"] = result[month_col].dt.quarter
    result["forecast_year"] = result[month_col].dt.year
    return result


def split_by_month(df, month_col="forecast_month", train_before="2026-04-01", valid_month="2026-04-01", test_month="2026-05-01"):
    """Стандартное временное разбиение: train до valid, valid следующий месяц, test следующий."""
    work = df.copy()
    work[month_col] = pd.to_datetime(work[month_col])
    train_before = pd.to_datetime(train_before)
    valid_month = pd.to_datetime(valid_month)
    test_month = pd.to_datetime(test_month)

    train = work[work[month_col] < train_before].copy()
    valid = work[work[month_col] == valid_month].copy()
    test = work[work[month_col] == test_month].copy()
    return train, valid, test


def save_split(train, valid, test, output_dir, segment, suffix="06_fs_selected"):
    """Сохраняет train/valid/test в формате csv."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"train_{segment}_{suffix}.csv"
    valid_path = output_dir / f"valid_{segment}_{suffix}.csv"
    test_path = output_dir / f"test_{segment}_{suffix}.csv"
    train.to_csv(train_path, index=False)
    valid.to_csv(valid_path, index=False)
    test.to_csv(test_path, index=False)
    return str(train_path), str(valid_path), str(test_path)


def make_pool(csv_path, cd_path):
    """Создаёт Pool из csv и cd-файла."""
    return Pool(data=csv_path, column_description=cd_path, delimiter=",", has_header=True)


def fit_regressor(train_pool, valid_pool, learning_rate=0.03, iterations=1000, depth=6, l2_leaf_reg=5, random_seed=42):
    """Единое обучение CatBoostRegressor."""
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=random_seed,
        early_stopping_rounds=50,
        verbose=100,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    return model


def fit_classifier(train_pool, valid_pool, learning_rate=0.03, iterations=1000, depth=6, l2_leaf_reg=5, random_seed=42):
    """Единое обучение CatBoostClassifier."""
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=random_seed,
        early_stopping_rounds=50,
        verbose=100,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    return model


def get_feature_importance_df(model, pool):
    """Возвращает feature importance CatBoost в виде DataFrame."""
    return pd.DataFrame({
        "feature": pool.get_feature_names(),
        "importance": model.get_feature_importance(pool),
    }).sort_values("importance", ascending=False)


def save_feature_importance(model, pool, output_path):
    """Сохраняет feature importance в csv."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fi = get_feature_importance_df(model, pool)
    fi.to_csv(output_path, index=False)
    return fi


def save_json(obj, output_path):
    """Сохраняет словарь/список в json."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_pickle(path):
    """Загружает pickle-объект."""
    with open(path, "rb") as f:
        return pickle.load(f)
