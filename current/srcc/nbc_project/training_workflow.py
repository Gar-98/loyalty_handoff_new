
"""
Единый train workflow для старых ноутбуков проекта NBC.

Зачем нужен модуль:
- сохранить исходную бизнес-логику train-ноутбуков;
- убрать повторяющиеся блоки подготовки split-ов, CatBoost Pool, обучения,
  метрик, сохранения моделей и feature importance;
- сделать ноутбуки запускаемыми через Run All сверху вниз.

Гранулярность данных и отбор фичей сохраняются: ноутбуки читают те же
*_fs_sel.csv файлы, которые были подготовлены в data_fm.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional, Literal
import json
import pickle
import time

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

try:
    from sklearn.isotonic import IsotonicRegression
except Exception:  # pragma: no cover
    IsotonicRegression = None

from .training_helpers import add_standard_time_features, SEASON_BASE_COLS
from .feature_selection import (
    normalize_feature_selection_config,
    select_features,
    save_selection_report,
)

TaskType = Literal["regression", "classification"]


@dataclass
class TrainJobConfig:
    """Описание одного сегмента обучения: T или C."""
    model_key: str
    task_type: TaskType
    input_path: str
    target_col: str
    output_split_dir: str
    model_output_dir: str
    model_filename: str
    ignore_cols: list[str]
    train_end_exclusive: str = "2026-05-01"
    valid_month: str = "2026-05-01"
    test_month: Optional[str] = "2026-06-01"
    month_col: str = "forecast_month"
    add_time_features: bool = True
    split_suffix: str = "06_fs_selected"
    legacy_test_suffix: Optional[str] = "fs_selected"
    iterations: int = 1000
    learning_rate: float = 0.03
    depth: int = 6
    l2_leaf_reg: float = 5.0
    random_seed: int = 42
    early_stopping_rounds: int = 50
    auto_class_weights: Optional[str] = None
    calibrate_isotonic: bool = False
    # None сохраняет текущую логику. Для dynamic FS передай dict, например
    # {"method": "psi_importance", "psi_threshold": 0.2, "top_n": 200}
    # или no-fixed-N: {"method": "psi_metric_rfe", "metric_rfe_rule": "one_se"}.
    feature_selection: Optional[dict] = None


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_month_start(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp()


def _available_columns(path: str | Path) -> list[str]:
    """Возвращает схему csv/parquet без чтения всего датасета."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл с обучающими данными: {path}")
    if path.is_dir() or path.suffix.lower() in {".parquet", ".pq"}:
        import pyarrow.dataset as ds
        return list(ds.dataset(str(path), format="parquet").schema.names)
    return list(pd.read_csv(path, nrows=0).columns)


def read_training_frame(path: str | Path, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Читает csv/parquet; columns позволяет после FS загрузить только нужные признаки."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл с обучающими данными: {path}")
    usecols = None
    if columns is not None:
        available = set(_available_columns(path))
        usecols = [c for c in columns if c in available]
    if path.is_dir() or path.suffix.lower() in {".parquet", ".pq"}:
        import pyarrow.dataset as ds
        dataset = ds.dataset(str(path), format="parquet")
        table = dataset.to_table(columns=usecols)
        return table.to_pandas()
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def sample_training_frame(
    path: str | Path,
    max_rows: int = 300_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Равномерная sample по всему source для dynamic feature selection.
    Для parquet сканирует батчами, но материализует только небольшую долю строк.
    """
    path = Path(path)
    if max_rows <= 0:
        raise ValueError("max_rows must be > 0")

    rng = np.random.default_rng(random_state)
    if path.is_dir() or path.suffix.lower() in {".parquet", ".pq"}:
        import pyarrow.dataset as ds
        dataset = ds.dataset(str(path), format="parquet")
        total_rows = max(int(dataset.count_rows()), 1)
        frac = min(1.0, max_rows / total_rows * 1.15)
        parts = []
        for batch in dataset.to_batches(batch_size=50_000):
            df_part = batch.to_pandas()
            if len(df_part) == 0:
                continue
            if frac < 1.0:
                mask = rng.random(len(df_part)) < frac
                df_part = df_part.loc[mask]
            if len(df_part):
                parts.append(df_part)
        if not parts:
            return pd.DataFrame(columns=dataset.schema.names)
        sample = pd.concat(parts, ignore_index=True)
    else:
        # Reservoir-like priority sample: равномерно по всему CSV без знания числа строк заранее.
        reservoir = None
        key_col = "__fs_sample_key__"
        for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
            chunk = chunk.copy()
            chunk[key_col] = rng.random(len(chunk))
            if reservoir is None:
                reservoir = chunk
            else:
                reservoir = pd.concat([reservoir, chunk], ignore_index=True)
            if len(reservoir) > max_rows:
                reservoir = reservoir.nsmallest(max_rows, key_col)
        if reservoir is None:
            sample = pd.DataFrame()
        else:
            sample = reservoir.drop(columns=[key_col]).reset_index(drop=True)

    if len(sample) > max_rows:
        sample = sample.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
    return sample


def _source_columns_for_selected(
    path: str | Path,
    selected_features: list[str],
    target_col: str,
    month_col: str,
    ignore_cols: Iterable[str],
    add_time_features_flag: bool,
) -> list[str]:
    """Определяет минимальный набор source columns для финального full-data train."""
    available = set(_available_columns(path))
    needed = [target_col, month_col, *list(ignore_cols or [])]
    needed += [c for c in selected_features if c in available]

    if add_time_features_flag:
        # *_season_adj вычисляются уже после чтения. Если такая фича выбрана, нужен её base column.
        for base in SEASON_BASE_COLS:
            if f"{base}_season_adj" in selected_features and base in available:
                needed.append(base)

    return [c for c in dict.fromkeys(needed) if c in available]


def choose_feature_cols(
    df: pd.DataFrame,
    target_col: str,
    month_col: str = "forecast_month",
    ignore_cols: Optional[Iterable[str]] = None,
    explicit_feature_cols: Optional[Iterable[str]] = None,
) -> list[str]:
    """Возвращает список признаков, не ломая старую fs_sel-логику."""
    if explicit_feature_cols is not None:
        return [c for c in explicit_feature_cols if c in df.columns and c != target_col]
    ignore = set(ignore_cols or [])
    ignore.update({target_col, month_col})
    return [c for c in df.columns if c not in ignore]


def prepare_frame(
    df: pd.DataFrame,
    target_col: str,
    month_col: str,
    ignore_cols: Iterable[str],
    add_time_features_flag: bool = True,
    fill_numeric_missing: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Приводит фрейм к формату для CatBoost:
    - forecast_month к month start;
    - добавляет сезонные/time features как в старых ноутбуках;
    - числовые object-колонки приводит к numeric, если это возможно;
    - categorical object-колонки оставляет строками и заполняет missing;
    - в current/static режиме числовые missing как раньше заполняются 0;
      dynamic FS может оставить NaN до prefilter, чтобы missing_share считался честно.
    """
    work = df.copy()
    if month_col not in work.columns:
        raise KeyError(f"В датасете нет month_col={month_col}")
    if target_col not in work.columns:
        raise KeyError(f"В датасете нет target_col={target_col}")

    work[month_col] = _to_month_start(work[month_col])
    if add_time_features_flag:
        work = add_standard_time_features(work, month_col=month_col)

    feature_cols = choose_feature_cols(
        work,
        target_col=target_col,
        month_col=month_col,
        ignore_cols=ignore_cols,
    )

    # target всегда числовой
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")

    cat_cols: list[str] = []
    for col in feature_cols:
        if pd.api.types.is_bool_dtype(work[col]):
            cat_cols.append(col)
            work[col] = work[col].astype("object").fillna("__MISSING__").astype(str)
        elif pd.api.types.is_object_dtype(work[col]) or str(work[col].dtype) == "category":
            converted = pd.to_numeric(work[col], errors="coerce")
            non_null_original = work[col].notna().sum()
            numeric_share = converted.notna().sum() / max(non_null_original, 1)
            if numeric_share > 0.95:
                work[col] = converted.astype("float64")
            else:
                cat_cols.append(col)
                work[col] = work[col].fillna("__MISSING__").astype(str)
        else:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    num_cols = [c for c in feature_cols if c not in cat_cols]
    if num_cols:
        work[num_cols] = work[num_cols].replace([np.inf, -np.inf], np.nan)
        if fill_numeric_missing:
            work[num_cols] = work[num_cols].fillna(0)
    work[target_col] = work[target_col].replace([np.inf, -np.inf], np.nan).fillna(0)
    return work, feature_cols, cat_cols


def split_frame(
    df: pd.DataFrame,
    month_col: str = "forecast_month",
    train_end_exclusive: str = "2026-05-01",
    valid_month: str = "2026-04-01",
    test_month: Optional[str] = "2026-06-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end_exclusive = pd.Timestamp(train_end_exclusive)
    valid_month = pd.Timestamp(valid_month)
    train = df[df[month_col] < train_end_exclusive].copy()
    valid = df[df[month_col] == valid_month].copy()
    if test_month is None:
        test = pd.DataFrame(columns=df.columns)
    else:
        test_month_ts = pd.Timestamp(test_month)
        test = df[df[month_col] == test_month_ts].copy()
    return train, valid, test


def save_splits(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str | Path,
    segment: str,
    suffix: str = "06_fs_selected",
    legacy_test_suffix: Optional[str] = "fs_selected",
) -> dict[str, str]:
    output_dir = _ensure_dir(output_dir)
    paths = {
        "train": output_dir / f"train_{segment}_{suffix}.csv",
        "valid": output_dir / f"valid_{segment}_{suffix}.csv",
        "test": output_dir / f"test_{segment}_{suffix}.csv",
    }
    train.to_csv(paths["train"], index=False)
    valid.to_csv(paths["valid"], index=False)
    test.to_csv(paths["test"], index=False)

    if legacy_test_suffix:
        legacy_test = output_dir / f"test_{segment}_{legacy_test_suffix}.csv"
        test.to_csv(legacy_test, index=False)
        paths["test_legacy"] = legacy_test
    return {k: str(v) for k, v in paths.items()}


def make_dataframe_pool(df: pd.DataFrame, target_col: str, feature_cols: list[str], cat_cols: list[str]) -> Pool:
    cat_idx = [feature_cols.index(c) for c in cat_cols if c in feature_cols]
    return Pool(df[feature_cols], df[target_col], cat_features=cat_idx)


def regression_metrics_dict(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    out = {
        "rows": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
    }
    out["r2"] = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    denom = np.sum(np.abs(y_true))
    out["wape"] = float(np.sum(np.abs(y_true - y_pred)) / denom) if denom != 0 else np.nan
    return out


def classification_metrics_dict(y_true, y_score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_score = np.asarray(y_score).reshape(-1)
    y_pred = (y_score >= threshold).astype(int)
    out = {
        "rows": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else np.nan,
        "avg_score": float(np.mean(y_score)) if len(y_score) else np.nan,
        "precision@0.5": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall@0.5": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1@0.5": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        out.update({
            "roc_auc": float(roc_auc_score(y_true, y_score)),
            "pr_auc": float(average_precision_score(y_true, y_score)),
            "logloss": float(log_loss(y_true, np.clip(y_score, 1e-6, 1 - 1e-6))),
            "brier": float(brier_score_loss(y_true, y_score)),
        })
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    else:
        out.update({"roc_auc": np.nan, "pr_auc": np.nan, "logloss": np.nan, "brier": np.nan})
    return out


def fit_model(config: TrainJobConfig, train_pool: Pool, valid_pool: Pool):
    common = dict(
        iterations=config.iterations,
        learning_rate=config.learning_rate,
        depth=config.depth,
        l2_leaf_reg=config.l2_leaf_reg,
        random_seed=config.random_seed,
        early_stopping_rounds=config.early_stopping_rounds,
        verbose=100,
        allow_writing_files=False,
    )
    if config.task_type == "classification":
        if config.auto_class_weights:
            common["auto_class_weights"] = config.auto_class_weights
        model = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", **common)
    else:
        model = CatBoostRegressor(loss_function="RMSE", eval_metric="RMSE", **common)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    return model


def predict_model(model, df: pd.DataFrame, feature_cols: list[str], task_type: TaskType) -> np.ndarray:
    if task_type == "classification":
        return model.predict_proba(df[feature_cols])[:, 1]
    return np.clip(model.predict(df[feature_cols]), 0, None)


def _feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    try:
        values = model.get_feature_importance()
        return pd.DataFrame({"feature": feature_cols, "importance": values}).sort_values("importance", ascending=False)
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])


def save_model_artifacts(
    model,
    config: TrainJobConfig,
    feature_cols: list[str],
    cat_cols: list[str],
    metrics: dict,
    calibrator=None,
) -> dict[str, str]:
    model_dir = _ensure_dir(config.model_output_dir)
    cbm_path = model_dir / config.model_filename
    pkl_path = model_dir / config.model_filename.replace(".cbm", ".pkl")
    meta_path = model_dir / config.model_filename.replace(".cbm", "_metadata.json")
    fi_path = model_dir / config.model_filename.replace(".cbm", "_feature_importance.csv")

    model.save_model(cbm_path)
    bundle = {
        "model": model,
        "calibrator": calibrator,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "target_col": config.target_col,
        "task_type": config.task_type,
        "config": asdict(config),
        "metrics": metrics,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(bundle, f)

    metadata = {
        "model_key": config.model_key,
        "task_type": config.task_type,
        "target_col": config.target_col,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "n_features": len(feature_cols),
        "metrics": metrics,
        "config": asdict(config),
        "cbm_path": str(cbm_path),
        "pkl_path": str(pkl_path),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    fi = _feature_importance(model, feature_cols)
    fi.to_csv(fi_path, index=False)
    return {"cbm": str(cbm_path), "pkl": str(pkl_path), "metadata": str(meta_path), "feature_importance": str(fi_path)}


def train_one_job(config: TrainJobConfig) -> dict:
    """Полный цикл: read -> prepare -> split -> train -> eval -> save."""
    t0 = time.time()
    print(f"\n===== {config.model_key} =====")
    print(f"Читаю данные: {config.input_path}")
    fs_cfg = normalize_feature_selection_config(config.feature_selection)
    selection_report = None
    selection_metadata = None
    selection_paths = {}

    if fs_cfg.method != "none":
        # 1) Dynamic FS делаем на ограниченной sample из полного source.
        # Test/OOT в selection не участвует.
        cutoff_info = (
            f"metric_rule={fs_cfg.metric_rfe_rule}" if "metric_rfe" in fs_cfg.method
            else f"permutation_ci={fs_cfg.permutation_ci_confidence:.0%}" if "permutation_ci" in fs_cfg.method
            else f"null_q={fs_cfg.null_importance_quantile:.0%}" if "null_importance" in fs_cfg.method
            else f"top_n={fs_cfg.top_n}"
        )
        print(
            f"dynamic feature selection: method={fs_cfg.method}, "
            f"sample_rows={fs_cfg.sample_rows:,}, {cutoff_info}, "
            f"psi_threshold={fs_cfg.psi_threshold}"
        )
        sample_raw = sample_training_frame(
            config.input_path,
            max_rows=fs_cfg.sample_rows,
            random_state=fs_cfg.sample_random_state,
        )
        print(f"feature-selection sample shape: {sample_raw.shape}")

        # Full modular sources contain several future targets. Dynamic FS must never
        # see any target_* except the target of the current model. Current/static
        # baseline is intentionally untouched because this block runs only for FS.
        dynamic_ignore_cols = list(config.ignore_cols)
        dynamic_ignore_cols += [
            c for c in sample_raw.columns
            if c != config.target_col and str(c).startswith("target_")
        ]
        dynamic_ignore_cols = list(dict.fromkeys(dynamic_ignore_cols))

        sample_df, candidate_features, sample_cat_cols = prepare_frame(
            sample_raw,
            target_col=config.target_col,
            month_col=config.month_col,
            ignore_cols=dynamic_ignore_cols,
            add_time_features_flag=config.add_time_features,
            fill_numeric_missing=False,
        )
        fs_train, fs_valid, _ = split_frame(
            sample_df,
            month_col=config.month_col,
            train_end_exclusive=config.train_end_exclusive,
            valid_month=config.valid_month,
            test_month=config.test_month,
        )
        if fs_train.empty or fs_valid.empty:
            raise ValueError(
                "Dynamic FS sample не содержит train/valid. "
                "Увеличь feature_selection.sample_rows или проверь split-месяцы."
            )
        selected_features, selection_report, selection_metadata = select_features(
            train=fs_train,
            valid=fs_valid,
            target_col=config.target_col,
            task_type=config.task_type,
            candidate_features=candidate_features,
            cat_cols=sample_cat_cols,
            month_col=config.month_col,
            config=fs_cfg,
        )
        selection_metadata.update({
            "model_key": config.model_key,
            "selection_train_rows": int(len(fs_train)),
            "selection_valid_rows": int(len(fs_valid)),
            "selection_train_months": [str(pd.Timestamp(x).date()) for x in sorted(fs_train[config.month_col].dropna().unique())],
            "selection_valid_months": [str(pd.Timestamp(x).date()) for x in sorted(fs_valid[config.month_col].dropna().unique())],
            "dynamic_ignore_cols": dynamic_ignore_cols,
        })
        selection_paths = save_selection_report(
            selection_report,
            selection_metadata,
            model_key=config.model_key,
            config=fs_cfg,
        )
        print(f"selected features: {len(selected_features)} / {len(candidate_features)}")
        print("feature selection artifacts:", selection_paths)

        del fs_train, fs_valid, sample_df, sample_raw
        gc.collect()

        # 2) После отбора читаем FULL dataset уже только по выбранным source columns.
        source_columns = _source_columns_for_selected(
            config.input_path,
            selected_features=selected_features,
            target_col=config.target_col,
            month_col=config.month_col,
            ignore_cols=dynamic_ignore_cols,
            add_time_features_flag=config.add_time_features,
        )
        raw = read_training_frame(config.input_path, columns=source_columns)
        print(f"full raw shape after column pruning: {raw.shape}")
        df, prepared_features, prepared_cat_cols = prepare_frame(
            raw,
            target_col=config.target_col,
            month_col=config.month_col,
            ignore_cols=dynamic_ignore_cols,
            add_time_features_flag=config.add_time_features,
            fill_numeric_missing=False,
        )
        feature_cols = [c for c in selected_features if c in df.columns]
        missing_selected = [c for c in selected_features if c not in df.columns]
        if missing_selected:
            print(f"WARNING: selected features absent after full read: {missing_selected[:20]}")
        cat_cols = [c for c in prepared_cat_cols if c in feature_cols]
        del raw
        gc.collect()
        if not feature_cols:
            raise ValueError("После full-data read не осталось выбранных признаков")
    else:
        # Текущий baseline: если input_path уже *_fs_sel.csv, поведение полностью прежнее.
        raw = read_training_frame(config.input_path)
        print(f"raw shape: {raw.shape}")
        df, feature_cols, cat_cols = prepare_frame(
            raw,
            target_col=config.target_col,
            month_col=config.month_col,
            ignore_cols=config.ignore_cols,
            add_time_features_flag=config.add_time_features,
        )

    print(f"features: {len(feature_cols)}, categorical: {len(cat_cols)}")

    train, valid, test = split_frame(
        df,
        month_col=config.month_col,
        train_end_exclusive=config.train_end_exclusive,
        valid_month=config.valid_month,
        test_month=config.test_month,
    )
    print(f"train/valid/test: {train.shape} / {valid.shape} / {test.shape}")
    del df
    gc.collect()
    if train.empty or valid.empty:
        raise ValueError("Train или valid пустой. Проверь train_end_exclusive/valid_month и forecast_month.")
    if config.test_month is not None and test.empty:
        raise ValueError(
            f"OOT/test пустой для test_month={config.test_month}. "
            "Benchmark не должен подменять независимый OOT validation-метриками."
        )

    segment = "T" if config.model_key.endswith("_T") else "C" if config.model_key.endswith("_C") else config.model_key
    split_paths = save_splits(
        train,
        valid,
        test,
        output_dir=config.output_split_dir,
        segment=segment,
        suffix=config.split_suffix,
        legacy_test_suffix=config.legacy_test_suffix,
    )
    print("split paths:", split_paths)

    train_pool = make_dataframe_pool(train, config.target_col, feature_cols, cat_cols)
    valid_pool = make_dataframe_pool(valid, config.target_col, feature_cols, cat_cols)
    model = fit_model(config, train_pool, valid_pool)

    pred_valid_raw = predict_model(model, valid, feature_cols, config.task_type)
    pred_test_raw = predict_model(model, test, feature_cols, config.task_type) if not test.empty else np.array([])

    calibrator = None
    pred_valid = pred_valid_raw
    pred_test = pred_test_raw
    if config.task_type == "classification" and config.calibrate_isotonic:
        if IsotonicRegression is None:
            print("IsotonicRegression недоступен, калибровка пропущена")
        elif valid[config.target_col].nunique() > 1:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(pred_valid_raw, valid[config.target_col].astype(int))
            pred_valid = calibrator.predict(pred_valid_raw)
            pred_test = calibrator.predict(pred_test_raw) if len(pred_test_raw) else pred_test_raw
        else:
            print("В valid только один класс, калибровка пропущена")

    if config.task_type == "classification":
        valid_metrics = classification_metrics_dict(valid[config.target_col], pred_valid)
        test_metrics = classification_metrics_dict(test[config.target_col], pred_test) if not test.empty else {}
    else:
        valid_metrics = regression_metrics_dict(valid[config.target_col], np.clip(pred_valid, 0, None))
        test_metrics = regression_metrics_dict(test[config.target_col], np.clip(pred_test, 0, None)) if not test.empty else {}

    metrics = {"valid": valid_metrics, "test": test_metrics}
    paths = save_model_artifacts(model, config, feature_cols, cat_cols, metrics, calibrator=calibrator)

    result_dir = _ensure_dir(Path(config.model_output_dir) / "predictions")
    valid_pred_df = valid[[c for c in ["cdi_id", config.month_col, config.target_col] if c in valid.columns]].copy()
    valid_pred_df["prediction"] = pred_valid
    valid_pred_df.to_csv(result_dir / f"{config.model_key}_valid_predictions.csv", index=False)
    if not test.empty:
        test_pred_df = test[[c for c in ["cdi_id", config.month_col, config.target_col] if c in test.columns]].copy()
        test_pred_df["prediction"] = pred_test
        test_pred_df.to_csv(result_dir / f"{config.model_key}_test_predictions.csv", index=False)

    print("valid metrics:", valid_metrics)
    print("test metrics:", test_metrics)
    print("artifacts:", paths)
    print(f"Готово за {(time.time() - t0) / 60:.1f} мин")
    return {
        "config": config,
        "model": model,
        "calibrator": calibrator,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "train": train,
        "valid": valid,
        "test": test,
        "pred_valid": pred_valid,
        "pred_test": pred_test,
        "metrics": metrics,
        "paths": {**paths, **selection_paths},
        "split_paths": split_paths,
        "feature_selection_report": selection_report,
        "feature_selection_metadata": selection_metadata,
        "elapsed_seconds": float(time.time() - t0),
    }


def run_jobs(configs: list[TrainJobConfig]) -> dict[str, dict]:
    results = {}
    for cfg in configs:
        results[cfg.model_key] = train_one_job(cfg)
    return results


def summarize_results(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for key, res in results.items():
        for split, metrics in res.get("metrics", {}).items():
            row = {"model_key": key, "split": split}
            row.update(metrics or {})
            rows.append(row)
    return pd.DataFrame(rows)


def run_optional_rolling(
    results: dict[str, dict],
    output_dir: str = "reports/rolling_backtest",
    min_train_months: int = 3,
    model_params: Optional[dict] = None,
):
    """Запускает rolling backtest по уже подготовленным фреймам из results."""
    from .rolling_backtest import run_rolling_backtest
    rolling = {}
    for key, res in results.items():
        cfg: TrainJobConfig = res["config"]
        df_full = pd.concat([res["train"], res["valid"], res["test"]], ignore_index=True)
        rolling[key] = run_rolling_backtest(
            df=df_full,
            target_col=cfg.target_col,
            task_type=cfg.task_type,
            model_key=key,
            output_dir=output_dir,
            month_col=cfg.month_col,
            ignore_cols=cfg.ignore_cols,
            feature_cols=res["feature_cols"],
            min_train_months=min_train_months,
            model_params=model_params or {
                "iterations": cfg.iterations,
                "learning_rate": cfg.learning_rate,
                "depth": cfg.depth,
                "l2_leaf_reg": cfg.l2_leaf_reg,
                "random_seed": cfg.random_seed,
                "early_stopping_rounds": cfg.early_stopping_rounds,
                "verbose": 100,
            },
        )
    return rolling


def run_optional_visuals(results: dict[str, dict], output_dir: str = "reports/training_visuals"):
    """
    Строит презентационные графики по каждой обученной модели:
    - распределение таргета train/valid/test;
    - bar chart метрик;
    - диагностику предсказаний;
    - ROC/PR/calibration/confusion matrix для classification;
    - feature importance.
    """
    from .training_visuals import make_regression_report, make_classification_report, plot_rolling_backtest_summary
    visuals = {}
    for key, res in results.items():
        cfg: TrainJobConfig = res["config"]
        try:
            split_targets = {
                "train": res["train"][cfg.target_col] if "train" in res and not res["train"].empty else None,
                "valid": res["valid"][cfg.target_col] if "valid" in res and not res["valid"].empty else None,
                "test": res["test"][cfg.target_col] if "test" in res and not res["test"].empty else None,
            }
            eval_df = res["test"] if "test" in res and not res["test"].empty else res["valid"]
            if cfg.task_type == "classification":
                y_score = res["pred_test"] if "pred_test" in res and len(res["pred_test"]) else res["pred_valid"]
                visuals[key] = make_classification_report(
                    model_key=key,
                    target_col=cfg.target_col,
                    y_true=eval_df[cfg.target_col],
                    y_score=y_score,
                    output_dir=output_dir,
                    split_targets=split_targets,
                    model=res["model"],
                    threshold=0.5,
                )
            else:
                y_pred = res["pred_test"] if "pred_test" in res and len(res["pred_test"]) else res["pred_valid"]
                visuals[key] = make_regression_report(
                    model_key=key,
                    target_col=cfg.target_col,
                    y_true=eval_df[cfg.target_col],
                    y_pred=y_pred,
                    output_dir=output_dir,
                    split_targets=split_targets,
                    model=res["model"],
                )
            # Если rolling уже был посчитан, дополнительно кладём heatmap в каталог visuals.
            try:
                plot_rolling_backtest_summary(key, rolling_dir="reports/rolling_backtest", output_dir=output_dir)
            except Exception:
                pass
        except Exception as exc:
            print(f"Не удалось построить visuals для {key}: {exc}")
    return visuals
