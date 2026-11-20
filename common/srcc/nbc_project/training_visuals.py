"""
Интерактивные Plotly-визуализации для train-ноутбуков.

Все графики сохраняются в HTML, поэтому в браузере/ноутбуке можно приближать,
двигать область, включать/выключать серии и смотреть hover-подсказки.

Модуль не обучает модели, а только строит и сохраняет графики:
- распределение таргета;
- метрики;
- fact vs prediction;
- residuals;
- ROC/PR/calibration/confusion matrix;
- feature importance;
- rolling/OOT heatmap.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as exc:
    raise ImportError(
        "Для интерактивных графиков нужен plotly. Установи: pip install plotly"
    ) from exc

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
    roc_curve,
    precision_recall_curve,
)

try:
    from sklearn.calibration import calibration_curve
except Exception:
    calibration_curve = None


PLOTLY_TEMPLATE = "plotly_white"
PLOTLY_RENDERER = None  # можно переопределить в ноутбуке: import plotly.io as pio; pio.renderers.default = "notebook_connected"
SHOW_PLOTLY_FIGURES = False  # True — показывать графики inline; False — только сохранять HTML


def _as_path_list(path_or_paths):
    """Приводит путь или список путей к списку Path."""
    if path_or_paths is None:
        return []
    if isinstance(path_or_paths, (str, Path)):
        return [Path(path_or_paths)]
    return [Path(p) for p in path_or_paths]


def _first_existing_path(path_or_paths):
    """Возвращает первый существующий путь из списка."""
    for path in _as_path_list(path_or_paths):
        if path.exists():
            return path
    return None


def _ensure_dir(path):
    """Создаёт директорию, если её ещё нет."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_series(y, name="target"):
    """Приводит y к pandas Series."""
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 1:
            return y.iloc[:, 0].rename(name)
        raise ValueError("Для y передан DataFrame с несколькими колонками")
    if isinstance(y, pd.Series):
        return y.rename(name)
    return pd.Series(np.asarray(y).reshape(-1), name=name)


def _read_target(path_or_paths, target_col):
    """Читает таргет из первого существующего csv/parquet файла."""
    path = _first_existing_path(path_or_paths)
    if path is None:
        return None
    if path.suffix.lower() in [".parquet", ".pq"]:
        df = pd.read_parquet(path, columns=[target_col])
    else:
        df = pd.read_csv(path, usecols=[target_col])
    return _to_series(df[target_col], name=target_col)


def _sample_for_plot(df: pd.DataFrame, max_points: int = 120_000, random_state: int = 42) -> pd.DataFrame:
    """Сэмплирует очень большие датафреймы только для scatter/hover-графиков."""
    if len(df) <= max_points:
        return df
    return df.sample(max_points, random_state=random_state)


def _save_plotly(fig, output_dir, filename, show=None):
    """Сохраняет Plotly-график в self-contained HTML и показывает его в ноутбуке."""
    output_dir = _ensure_dir(output_dir)
    filename = filename.replace(".png", ".html")
    if not filename.endswith(".html"):
        filename = f"{filename}.html"
    path = output_dir / filename

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        hovermode="closest",
        margin=dict(l=40, r=30, t=70, b=50),
    )
    fig.write_html(str(path), include_plotlyjs=True, full_html=True)

    if show is None:
        show = SHOW_PLOTLY_FIGURES
    if show:
        try:
            fig.show(renderer=PLOTLY_RENDERER)
        except Exception:
            # В некоторых Jupyter-окружениях renderer может быть не настроен.
            pass
    return str(path)


def regression_metrics_dict(y_true, y_pred):
    """Считает базовые метрики регрессии."""
    y_true = np.asarray(_to_series(y_true)).reshape(-1)
    y_pred = np.asarray(_to_series(y_pred)).reshape(-1)
    metrics = {
        "rows": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "medae": float(median_absolute_error(y_true, y_pred)),
    }
    if len(y_true) > 1:
        metrics["r2"] = float(r2_score(y_true, y_pred))
    else:
        metrics["r2"] = np.nan
    denom = np.sum(np.abs(y_true))
    metrics["wape"] = float(np.sum(np.abs(y_true - y_pred)) / denom) if denom != 0 else np.nan
    return metrics


def classification_metrics_dict(y_true, y_score, threshold=0.5):
    """Считает базовые метрики бинарной классификации."""
    y_true = np.asarray(_to_series(y_true)).reshape(-1).astype(int)
    y_score = np.asarray(_to_series(y_score)).reshape(-1)
    y_pred = (y_score >= threshold).astype(int)
    metrics = {
        "rows": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else np.nan,
        "avg_score": float(np.mean(y_score)) if len(y_score) else np.nan,
        "brier": float(brier_score_loss(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        metrics["logloss"] = float(log_loss(y_true, np.clip(y_score, 1e-6, 1 - 1e-6)))
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
        metrics["logloss"] = np.nan
    return metrics


def save_metrics(metrics, output_dir):
    """Сохраняет метрики в json и csv."""
    output_dir = _ensure_dir(output_dir)
    clean_metrics = {
        k: (None if pd.isna(v) else float(v) if isinstance(v, (int, float, np.number)) else v)
        for k, v in metrics.items()
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(clean_metrics, f, ensure_ascii=False, indent=2)
    pd.DataFrame([clean_metrics]).to_csv(output_dir / "metrics.csv", index=False)


def plot_metrics_bar(metrics, output_dir, title="Метрики модели"):
    """Строит интерактивный bar chart метрик."""
    items = []
    for key, value in metrics.items():
        if key == "rows":
            continue
        if isinstance(value, (int, float, np.number)) and not pd.isna(value):
            items.append((key, float(value)))
    if not items:
        return None

    df = pd.DataFrame(items, columns=["metric", "value"])
    fig = px.bar(
        df,
        x="metric",
        y="value",
        text="value",
        title=title,
        labels={"metric": "Метрика", "value": "Значение"},
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(xaxis_tickangle=-35)
    return _save_plotly(fig, output_dir, "metrics_bar.html")


def plot_target_distribution_by_split(split_targets, output_dir, target_col, model_key):
    """Строит интерактивное распределение таргета по train/valid/test."""
    frames = []
    for split_name, y in split_targets.items():
        if y is None:
            continue
        s = _to_series(y, name=target_col).dropna()
        if len(s) == 0:
            continue
        frames.append(pd.DataFrame({target_col: s.values, "split": f"{split_name}, n={len(s):,}"}))
    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    fig = px.histogram(
        df,
        x=target_col,
        color="split",
        nbins=60,
        barmode="overlay",
        opacity=0.55,
        title=f"{model_key}: распределение таргета {target_col}",
        labels={target_col: target_col, "count": "Количество"},
    )
    return _save_plotly(fig, output_dir, "target_distribution.html")


def plot_target_box_by_split(split_targets, output_dir, target_col, model_key):
    """Строит интерактивный boxplot таргета по split-ам."""
    frames = []
    for split_name, y in split_targets.items():
        if y is None:
            continue
        s = _to_series(y, name=target_col).dropna()
        if len(s) == 0:
            continue
        frames.append(pd.DataFrame({target_col: s.values, "split": split_name}))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    fig = px.box(
        df,
        x="split",
        y=target_col,
        points=False,
        title=f"{model_key}: boxplot таргета",
        labels={"split": "Split", target_col: target_col},
    )
    return _save_plotly(fig, output_dir, "target_boxplot.html")


def plot_prediction_scatter(y_true, y_pred, output_dir, model_key):
    """Строит интерактивный scatter fact vs prediction для регрессии."""
    y_true = _to_series(y_true, name="y_true")
    y_pred = _to_series(y_pred, name="y_pred")
    n = min(len(y_true), len(y_pred))
    df = pd.DataFrame({"Факт": y_true.iloc[:n].values, "Прогноз": y_pred.iloc[:n].values})
    df = _sample_for_plot(df)

    min_v = np.nanmin([df["Факт"].min(), df["Прогноз"].min()])
    max_v = np.nanmax([df["Факт"].max(), df["Прогноз"].max()])

    fig = px.scatter(
        df,
        x="Факт",
        y="Прогноз",
        opacity=0.35,
        title=f"{model_key}: факт vs прогноз",
    )
    fig.update_traces(marker=dict(size=4))
    fig.add_trace(
        go.Scatter(
            x=[min_v, max_v],
            y=[min_v, max_v],
            mode="lines",
            name="Идеально",
            line=dict(dash="dash"),
        )
    )
    return _save_plotly(fig, output_dir, "fact_vs_prediction.html")


def plot_residuals(y_true, y_pred, output_dir, model_key):
    """Строит интерактивное распределение остатков для регрессии."""
    y_true = _to_series(y_true, name="y_true")
    y_pred = _to_series(y_pred, name="y_pred")
    n = min(len(y_true), len(y_pred))
    residuals = y_true.iloc[:n].values - y_pred.iloc[:n].values
    df = pd.DataFrame({"residual": residuals})
    fig = px.histogram(
        df,
        x="residual",
        nbins=80,
        title=f"{model_key}: распределение ошибок",
        labels={"residual": "Факт - прогноз", "count": "Количество"},
    )
    fig.add_vline(x=0, line_dash="dash")
    return _save_plotly(fig, output_dir, "residuals_distribution.html")


def plot_absolute_error_quantiles(y_true, y_pred, output_dir, model_key):
    """Строит интерактивные квантили абсолютной ошибки."""
    y_true = _to_series(y_true, name="y_true")
    y_pred = _to_series(y_pred, name="y_pred")
    n = min(len(y_true), len(y_pred))
    abs_err = np.abs(y_true.iloc[:n].values - y_pred.iloc[:n].values)
    q = pd.Series(abs_err).quantile([0.5, 0.75, 0.9, 0.95, 0.99])
    df = pd.DataFrame({"quantile": [str(x) for x in q.index], "abs_error": q.values})
    fig = px.bar(
        df,
        x="quantile",
        y="abs_error",
        text="abs_error",
        title=f"{model_key}: квантили абсолютной ошибки",
        labels={"quantile": "Квантиль", "abs_error": "Абсолютная ошибка"},
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    return _save_plotly(fig, output_dir, "absolute_error_quantiles.html")


def plot_probability_distribution(y_true, y_score, output_dir, model_key):
    """Строит интерактивное распределение вероятностей по классам."""
    y_true = _to_series(y_true, name="y_true").astype(int)
    y_score = _to_series(y_score, name="score")
    n = min(len(y_true), len(y_score))
    df = pd.DataFrame({"target": y_true.iloc[:n].values.astype(str), "score": y_score.iloc[:n].values})
    fig = px.histogram(
        df,
        x="score",
        color="target",
        nbins=50,
        barmode="overlay",
        opacity=0.55,
        title=f"{model_key}: распределение score по классам",
        labels={"score": "Score / probability", "count": "Количество", "target": "target"},
    )
    return _save_plotly(fig, output_dir, "probability_distribution.html")


def plot_roc_curve(y_true, y_score, output_dir, model_key):
    """Строит интерактивную ROC-кривую."""
    y_true = _to_series(y_true, name="y_true").astype(int)
    y_score = _to_series(y_score, name="score")
    if y_true.nunique() < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC-AUC = {auc_value:.4f}"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
    fig.update_layout(
        title=f"{model_key}: ROC-кривая",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    return _save_plotly(fig, output_dir, "roc_curve.html")


def plot_pr_curve(y_true, y_score, output_dir, model_key):
    """Строит интерактивную Precision-Recall кривую."""
    y_true = _to_series(y_true, name="y_true").astype(int)
    y_score = _to_series(y_score, name="score")
    if y_true.nunique() < 2:
        return None
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    base_rate = y_true.mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"PR-AUC = {pr_auc:.4f}"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[base_rate, base_rate], mode="lines", name=f"Base rate = {base_rate:.4f}", line=dict(dash="dash")))
    fig.update_layout(
        title=f"{model_key}: Precision-Recall",
        xaxis_title="Recall",
        yaxis_title="Precision",
    )
    return _save_plotly(fig, output_dir, "pr_curve.html")


def plot_calibration(y_true, y_score, output_dir, model_key, n_bins=10):
    """Строит интерактивную calibration curve."""
    if calibration_curve is None:
        return None
    y_true = _to_series(y_true, name="y_true").astype(int)
    y_score = _to_series(y_score, name="score")
    if y_true.nunique() < 2:
        return None
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="quantile")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name="Модель"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Идеальная калибровка", line=dict(dash="dash")))
    fig.update_layout(
        title=f"{model_key}: калибровка вероятностей",
        xaxis_title="Средний прогноз в бакете",
        yaxis_title="Фактическая доля класса 1",
    )
    return _save_plotly(fig, output_dir, "calibration_curve.html")


def plot_confusion_matrix(y_true, y_score, output_dir, model_key, threshold=0.5):
    """Строит интерактивную confusion matrix для выбранного threshold."""
    y_true = _to_series(y_true, name="y_true").astype(int)
    y_score = _to_series(y_score, name="score")
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = px.imshow(
        cm,
        text_auto=True,
        x=["pred 0", "pred 1"],
        y=["true 0", "true 1"],
        title=f"{model_key}: confusion matrix, threshold={threshold}",
        labels=dict(x="Прогноз", y="Факт", color="Количество"),
    )
    return _save_plotly(fig, output_dir, "confusion_matrix.html")


def plot_feature_importance(model, output_dir, model_key, top_n=30):
    """Строит интерактивный feature importance из обученной CatBoost-модели."""
    if model is None or not hasattr(model, "get_feature_importance"):
        return None
    try:
        importance = model.get_feature_importance()
        names = getattr(model, "feature_names_", None)
        if names is None or len(names) != len(importance):
            names = [f"feature_{i}" for i in range(len(importance))]
        fi = pd.DataFrame({"feature": names, "importance": importance})
        fi = fi.sort_values("importance", ascending=False)
        output_dir = _ensure_dir(output_dir)
        fi.to_csv(output_dir / "feature_importance.csv", index=False)
        fi_top = fi.head(top_n).sort_values("importance", ascending=True)
        fig = px.bar(
            fi_top,
            x="importance",
            y="feature",
            orientation="h",
            title=f"{model_key}: top-{top_n} feature importance",
            labels={"importance": "Importance", "feature": "Feature"},
        )
        fig.update_layout(height=max(500, len(fi_top) * 24))
        return _save_plotly(fig, output_dir, "feature_importance_top30.html")
    except Exception as exc:
        print(f"Не удалось построить feature importance для {model_key}: {exc}")
        return None


def _prepare_rolling_heatmap(path):
    """Читает heatmap.csv из wide или long формата и возвращает числовую pivot-таблицу."""
    raw = pd.read_csv(path)
    if raw.empty:
        return raw

    # Long-формат из rolling_backtest.py:
    # train_end_month, eval_month, ..., rmse/roc_auc/...
    long_index_candidates = ["train_end_month", "train_end", "train_month"]
    long_col_candidates = ["eval_month", "test_month", "valid_month", "month"]
    metric_candidates = ["rmse", "roc_auc", "mae", "wape", "r2", "pr_auc", "logloss", "f1", "precision", "recall", "metric_value", "value"]

    index_col = next((c for c in long_index_candidates if c in raw.columns), None)
    columns_col = next((c for c in long_col_candidates if c in raw.columns), None)
    value_col = next((c for c in metric_candidates if c in raw.columns), None)

    if index_col and columns_col and value_col:
        pivot = raw.pivot_table(index=index_col, columns=columns_col, values=value_col, aggfunc="mean")
        return pivot.sort_index().sort_index(axis=1)

    # Wide-формат: первая колонка может быть индексом/датой, остальные метрики.
    if raw.columns[0].lower() in ["unnamed: 0", "index", "train_end", "train_end_month"] or not pd.api.types.is_numeric_dtype(raw.iloc[:, 0]):
        maybe_index = raw.iloc[:, 0].astype(str)
        wide = raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        wide.index = maybe_index
    else:
        wide = raw.apply(pd.to_numeric, errors="coerce")

    wide = wide.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return wide


def plot_rolling_heatmap(heatmap_path, output_dir, model_key):
    """Строит интерактивную heatmap rolling/OOT метрик из сохранённого heatmap.csv."""
    path = _first_existing_path(heatmap_path)
    if path is None:
        return None
    heatmap = _prepare_rolling_heatmap(path)
    if heatmap.empty:
        return None

    fig = px.imshow(
        heatmap,
        text_auto=".3f",
        aspect="auto",
        title=f"{model_key}: rolling/OOT heatmap",
        labels=dict(x="OOT период", y="Train end", color="Metric"),
    )
    fig.update_layout(height=max(500, heatmap.shape[0] * 35), width=max(850, heatmap.shape[1] * 80))
    return _save_plotly(fig, output_dir, "rolling_heatmap.html")


def make_regression_report(
    model_key,
    y_true,
    y_pred,
    output_dir="reports/training_visuals",
    split_targets=None,
    target_col="target",
    model=None,
):
    """Собирает полный набор интерактивных графиков для регрессионной модели."""
    model_dir = _ensure_dir(Path(output_dir) / model_key)
    metrics = regression_metrics_dict(y_true, y_pred)
    save_metrics(metrics, model_dir)
    plot_metrics_bar(metrics, model_dir, title=f"{model_key}: метрики регрессии")
    if split_targets:
        plot_target_distribution_by_split(split_targets, model_dir, target_col, model_key)
        plot_target_box_by_split(split_targets, model_dir, target_col, model_key)
    plot_prediction_scatter(y_true, y_pred, model_dir, model_key)
    plot_residuals(y_true, y_pred, model_dir, model_key)
    plot_absolute_error_quantiles(y_true, y_pred, model_dir, model_key)
    plot_feature_importance(model, model_dir, model_key)
    print(f"Интерактивные Plotly-графики сохранены в: {model_dir}")
    return metrics


def make_classification_report(
    model_key,
    y_true,
    y_score,
    output_dir="reports/training_visuals",
    split_targets=None,
    target_col="target",
    model=None,
    threshold=0.5,
):
    """Собирает полный набор интерактивных графиков для бинарной модели."""
    model_dir = _ensure_dir(Path(output_dir) / model_key)
    metrics = classification_metrics_dict(y_true, y_score, threshold=threshold)
    save_metrics(metrics, model_dir)
    plot_metrics_bar(metrics, model_dir, title=f"{model_key}: метрики классификации")
    if split_targets:
        plot_target_distribution_by_split(split_targets, model_dir, target_col, model_key)
        plot_target_box_by_split(split_targets, model_dir, target_col, model_key)
    plot_probability_distribution(y_true, y_score, model_dir, model_key)
    plot_roc_curve(y_true, y_score, model_dir, model_key)
    plot_pr_curve(y_true, y_score, model_dir, model_key)
    plot_calibration(y_true, y_score, model_dir, model_key)
    plot_confusion_matrix(y_true, y_score, model_dir, model_key, threshold=threshold)
    plot_feature_importance(model, model_dir, model_key)
    print(f"Интерактивные Plotly-графики сохранены в: {model_dir}")
    return metrics


def make_regression_report_from_files(
    model_key,
    target_col,
    y_pred,
    test_path,
    train_path=None,
    valid_path=None,
    output_dir="reports/training_visuals",
    model=None,
):
    """Собирает регрессионный отчёт, читая таргеты из файлов train/valid/test."""
    y_test = _read_target(test_path, target_col)
    if y_test is None:
        raise FileNotFoundError(f"Не найден test-файл для {model_key}: {test_path}")
    split_targets = {
        "train": _read_target(train_path, target_col),
        "valid": _read_target(valid_path, target_col),
        "test": y_test,
    }
    return make_regression_report(
        model_key=model_key,
        y_true=y_test,
        y_pred=y_pred,
        output_dir=output_dir,
        split_targets=split_targets,
        target_col=target_col,
        model=model,
    )


def make_classification_report_from_files(
    model_key,
    target_col,
    y_score,
    test_path,
    train_path=None,
    valid_path=None,
    output_dir="reports/training_visuals",
    model=None,
    threshold=0.5,
):
    """Собирает классификационный отчёт, читая таргеты из файлов train/valid/test."""
    y_test = _read_target(test_path, target_col)
    if y_test is None:
        raise FileNotFoundError(f"Не найден test-файл для {model_key}: {test_path}")
    split_targets = {
        "train": _read_target(train_path, target_col),
        "valid": _read_target(valid_path, target_col),
        "test": y_test,
    }
    return make_classification_report(
        model_key=model_key,
        y_true=y_test,
        y_score=y_score,
        output_dir=output_dir,
        split_targets=split_targets,
        target_col=target_col,
        model=model,
        threshold=threshold,
    )


def plot_rolling_backtest_summary(model_key, rolling_dir="reports/rolling_backtest", output_dir="reports/training_visuals"):
    """Кладёт интерактивную rolling heatmap в общий каталог визуализаций."""
    model_dir = _ensure_dir(Path(output_dir) / model_key)
    heatmap_path = Path(rolling_dir) / model_key / "heatmap.csv"
    return plot_rolling_heatmap(heatmap_path, model_dir, model_key)
