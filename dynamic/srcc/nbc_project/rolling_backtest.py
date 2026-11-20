"""
Rolling / walk-forward backtest.

Идея проверки:
- train: все месяцы до train_end включительно;
- valid: следующий месяц после train_end;
- test: через один месяц после valid (valid + 2 календарных месяца);
- OOT: все будущие месяцы после train_end, чтобы построить heatmap качества.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
)


def _to_month_start(values):
    """Приводит даты к первому дню месяца."""
    return pd.to_datetime(values).dt.to_period("M").dt.to_timestamp()


def _safe_rmse(y_true, y_pred):
    """RMSE."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _safe_regression_metrics(y_true, y_pred):
    """Метрики регрессии для одного периода."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    result = {
        "rows": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": _safe_rmse(y_true, y_pred),
    }
    if len(y_true) > 1:
        result["r2"] = float(r2_score(y_true, y_pred))
    else:
        result["r2"] = np.nan
    return result


def _safe_classification_metrics(y_true, p_pred):
    """Метрики бинарной классификации для одного периода."""
    y_true = np.asarray(y_true).reshape(-1)
    p_pred = np.asarray(p_pred).reshape(-1)
    result = {
        "rows": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else np.nan,
        "avg_pred": float(np.mean(p_pred)) if len(p_pred) else np.nan,
    }

    if len(np.unique(y_true)) > 1:
        result["roc_auc"] = float(roc_auc_score(y_true, p_pred))
        result["pr_auc"] = float(average_precision_score(y_true, p_pred))
        result["logloss"] = float(log_loss(y_true, np.clip(p_pred, 1e-6, 1 - 1e-6)))
    else:
        result["roc_auc"] = np.nan
        result["pr_auc"] = np.nan
        result["logloss"] = np.nan
    return result


def _build_feature_cols(df, target_col, month_col, ignore_cols=None, feature_cols=None):
    """
    Собирает список признаков.

    Если feature_cols передан явно, используем его. Иначе удаляем таргет,
    месяц и служебные колонки.
    """
    if feature_cols is not None:
        return [c for c in feature_cols if c in df.columns and c != target_col]

    ignore = set(ignore_cols or [])
    ignore.update({target_col, month_col})
    return [c for c in df.columns if c not in ignore]


def _make_pool(frame, target_col, feature_cols, task_type):
    """Создаёт CatBoost Pool из DataFrame."""
    x = frame[feature_cols]
    y = frame[target_col]

    cat_features = [
        i
        for i, col in enumerate(feature_cols)
        if str(x[col].dtype) in {"object", "category", "bool"}
    ]
    return Pool(x, y, cat_features=cat_features)


def _fit_model(train_df, valid_df, target_col, feature_cols, task_type, model_params=None):
    """Обучает CatBoost для одного rolling-разбиения."""
    model_params = dict(model_params or {})
    common_params = {
        "iterations": model_params.pop("iterations", 1000),
        "learning_rate": model_params.pop("learning_rate", 0.03),
        "depth": model_params.pop("depth", 6),
        "l2_leaf_reg": model_params.pop("l2_leaf_reg", 5),
        "random_seed": model_params.pop("random_seed", 42),
        "early_stopping_rounds": model_params.pop("early_stopping_rounds", 50),
        "verbose": model_params.pop("verbose", 100),
        "allow_writing_files": model_params.pop("allow_writing_files", False),
    }
    common_params.update(model_params)

    train_pool = _make_pool(train_df, target_col, feature_cols, task_type)
    valid_pool = _make_pool(valid_df, target_col, feature_cols, task_type)

    if task_type == "classification":
        model = CatBoostClassifier(
            loss_function=common_params.pop("loss_function", "Logloss"),
            eval_metric=common_params.pop("eval_metric", "AUC"),
            **common_params,
        )
    else:
        model = CatBoostRegressor(
            loss_function=common_params.pop("loss_function", "RMSE"),
            eval_metric=common_params.pop("eval_metric", "RMSE"),
            **common_params,
        )

    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    return model


def _predict(model, eval_df, feature_cols, task_type):
    """Считает прогноз модели."""
    if task_type == "classification":
        return model.predict_proba(eval_df[feature_cols])[:, 1]
    return model.predict(eval_df[feature_cols])


def _score_period(model, eval_df, target_col, feature_cols, task_type):
    """Считает метрики на одном eval-периоде."""
    if eval_df.empty:
        return None
    pred = _predict(model, eval_df, feature_cols, task_type)
    if task_type == "classification":
        return _safe_classification_metrics(eval_df[target_col], pred)
    return _safe_regression_metrics(eval_df[target_col], pred)


def _plot_heatmap(heatmap_df, output_path, metric_col):
    """Сохраняет интерактивную Plotly heatmap rolling/OOT метрики."""
    if heatmap_df.empty:
        return None

    pivot = heatmap_df.pivot_table(
        index="train_end_month",
        columns="eval_month",
        values=metric_col,
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)

    if pivot.empty:
        return None

    import plotly.express as px

    # Читаемые подписи месяцев, но значения остаются pivot-таблицей.
    pivot_display = pivot.copy()
    try:
        pivot_display.index = [pd.Timestamp(x).strftime("%Y-%m") for x in pivot_display.index]
        pivot_display.columns = [pd.Timestamp(x).strftime("%Y-%m") for x in pivot_display.columns]
    except Exception:
        pass

    fig = px.imshow(
        pivot_display,
        text_auto=".3f",
        aspect="auto",
        title=f"Rolling/OOT backtest: {metric_col}",
        labels=dict(x="Месяц оценки", y="Последний месяц train", color=metric_col),
    )
    fig.update_layout(
        template="plotly_white",
        height=max(500, 35 * len(pivot_display.index)),
        width=max(850, 85 * len(pivot_display.columns)),
        margin=dict(l=40, r=30, t=70, b=50),
    )

    output_path = Path(output_path).with_suffix(".html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
    return str(output_path)


def run_rolling_backtest(
    df,
    target_col,
    task_type,
    model_key,
    output_dir,
    month_col="forecast_month",
    ignore_cols=None,
    feature_cols=None,
    min_train_months=3,
    model_params=None,
    score_all_future=True,
):
    """
    Запускает rolling backtest для одной модели.

    Параметры:
    df — датасет с признаками, месяцем и таргетом;
    target_col — колонка таргета;
    task_type — "regression" или "classification";
    model_key — имя модели для артефактов;
    output_dir — куда сохранять metrics.csv, heatmap.csv, heatmap.png;
    month_col — колонка месяца;
    ignore_cols — служебные колонки, которые не должны быть фичами;
    feature_cols — явный список фичей, если он уже собран в ноутбуке;
    min_train_months — минимальное число месяцев в train перед первым split;
    score_all_future — считать качество на всех будущих месяцах для heatmap.
    """
    if task_type not in {"regression", "classification"}:
        raise ValueError("task_type должен быть 'regression' или 'classification'")

    work = df.copy()
    work[month_col] = _to_month_start(work[month_col])
    work = work.dropna(subset=[month_col, target_col])

    features = _build_feature_cols(
        work,
        target_col=target_col,
        month_col=month_col,
        ignore_cols=ignore_cols,
        feature_cols=feature_cols,
    )

    months = [pd.Timestamp(x) for x in sorted(work[month_col].dropna().unique())]
    if len(months) < min_train_months + 3:
        raise ValueError(
            f"Недостаточно месяцев для rolling backtest: {len(months)}. "
            f"Нужно минимум {min_train_months + 3}: train + valid + пропущенный месяц + test."
        )

    output_dir = Path(output_dir) / model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    heatmap_rows = []

    # Последний доступный train_end должен оставлять:
    # 1) valid — следующий календарный месяц после train_end;
    # 2) test — через один месяц после valid, то есть valid + 2 месяца.
    # Пример: valid = 2026-03-01, test = 2026-05-01.
    months_set = set(months)
    for train_end_idx in range(min_train_months - 1, len(months)):
        train_months = months[: train_end_idx + 1]
        train_end_month = pd.Timestamp(months[train_end_idx])
        valid_month = train_end_month + pd.DateOffset(months=1)
        test_month = valid_month + pd.DateOffset(months=2)

        if valid_month not in months_set or test_month not in months_set:
            continue

        train_df = work[work[month_col].isin(train_months)].copy()
        valid_df = work[work[month_col] == valid_month].copy()
        test_df = work[work[month_col] == test_month].copy()

        if train_df.empty or valid_df.empty or test_df.empty:
            continue

        model = _fit_model(
            train_df=train_df,
            valid_df=valid_df,
            target_col=target_col,
            feature_cols=features,
            task_type=task_type,
            model_params=model_params,
        )

        for split_name, split_df, split_month in [
            ("valid", valid_df, valid_month),
            ("test", test_df, test_month),
        ]:
            score = _score_period(model, split_df, target_col, features, task_type)
            if score is None:
                continue
            row = {
                "model_key": model_key,
                "split": split_name,
                "train_end_month": pd.Timestamp(train_end_month).strftime("%Y-%m-01"),
                "eval_month": pd.Timestamp(split_month).strftime("%Y-%m-01"),
                "n_train_months": int(len(train_months)),
                "n_train_rows": int(len(train_df)),
                "n_eval_rows": int(len(split_df)),
                "n_features": int(len(features)),
            }
            row.update(score)
            metrics_rows.append(row)

        eval_months = months[train_end_idx + 1 :] if score_all_future else [valid_month, test_month]
        for eval_month in eval_months:
            eval_df = work[work[month_col] == eval_month].copy()
            score = _score_period(model, eval_df, target_col, features, task_type)
            if score is None:
                continue
            row = {
                "model_key": model_key,
                "train_end_month": pd.Timestamp(train_end_month).strftime("%Y-%m-01"),
                "eval_month": pd.Timestamp(eval_month).strftime("%Y-%m-01"),
                "n_train_months": int(len(train_months)),
                "n_train_rows": int(len(train_df)),
                "n_eval_rows": int(len(eval_df)),
                "n_features": int(len(features)),
            }
            row.update(score)
            heatmap_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    heatmap_df = pd.DataFrame(heatmap_rows)

    metrics_path = output_dir / "metrics.csv"
    heatmap_path = output_dir / "heatmap.csv"
    metadata_path = output_dir / "metadata.json"

    metrics_df.to_csv(metrics_path, index=False)
    heatmap_df.to_csv(heatmap_path, index=False)

    main_metric = "roc_auc" if task_type == "classification" else "rmse"
    heatmap_html = output_dir / "heatmap.html"
    if main_metric in heatmap_df.columns:
        heatmap_html = _plot_heatmap(heatmap_df, heatmap_html, main_metric) or heatmap_html

    metadata = {
        "model_key": model_key,
        "target_col": target_col,
        "task_type": task_type,
        "month_col": month_col,
        "min_train_months": min_train_months,
        "score_all_future": score_all_future,
        "n_features": len(features),
        "feature_cols": features,
        "main_metric": main_metric,
        "metrics_path": str(metrics_path),
        "heatmap_path": str(heatmap_path),
        "heatmap_html": str(heatmap_html),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Rolling backtest для {model_key} готов")
    print(f"Метрики: {metrics_path}")
    print(f"Heatmap csv: {heatmap_path}")
    print(f"Heatmap html: {heatmap_html}")

    return {
        "metrics": metrics_df,
        "heatmap": heatmap_df,
        "metadata": metadata,
    }
