"""Training workflow for the two models from 52.1.train_pipeline.ipynb.

The module preserves the model-specific logic of the original notebook:
- cashback regression is trained only on the connected-client dataset and predictions are clipped to [0, 5000];
- loyalty connection keeps a client-level 50/50 split of the validation month into
  ``valid_model`` (early stopping / feature selection) and ``valid_calib`` (isotonic calibration).

v6 uses the modular full-feature candidate for dynamic selection while keeping
``feature_selection=None`` as the exact static-feature baseline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import gc
import joblib
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from catboost import Pool, CatBoostRegressor, CatBoostClassifier

from .train_pipeline_helpers import get_df
from .training_workflow import (
    prepare_frame,
    regression_metrics_dict,
    classification_metrics_dict,
    save_model_artifacts,
    TrainJobConfig,
    sample_training_frame,
    read_training_frame,
    split_frame,
    _source_columns_for_selected,
    _available_columns,
)
from .feature_selection import (
    normalize_feature_selection_config,
    select_features,
    save_selection_report,
)


# Full feature mart contains labels / technical fields that were absent from the old static lists.
# They must never become dynamic candidate features.
COMMON_DYNAMIC_EXCLUDE = {
    "period",
    "period_dt",
    "rn",
    "n_treat",
    "target_flag",
    "delta_bin",
}


def _dynamic_ignore_cols(data_path: str, target_col: str, ignore_cols: Iterable[str]) -> list[str]:
    """Exclude all other target_* columns and known technical/future-label fields from dynamic FS."""
    ignore = list(dict.fromkeys(list(ignore_cols or []) + list(COMMON_DYNAMIC_EXCLUDE)))
    try:
        available = _available_columns(data_path)
    except Exception:
        available = []
    for col in available:
        if col != target_col and str(col).startswith("target_"):
            ignore.append(col)
    return list(dict.fromkeys(ignore))


def _prepare_static_or_dynamic(
    *,
    data_path: str,
    cfg: TrainJobConfig,
    train_start: Optional[str] = None,
    selection_valid_clients: Optional[Iterable] = None,
) -> dict:
    """Prepare full frame and feature list; dynamic selection sees train + selection-valid only.

    For loyalty, ``selection_valid_clients`` is the valid_model half. This intentionally keeps
    valid_calib untouched by feature selection, so it remains an honest calibration/evaluation split.
    """
    fs_cfg = normalize_feature_selection_config(cfg.feature_selection)
    selection_report = None
    selection_metadata = None
    selection_paths: dict[str, str] = {}

    if fs_cfg.method == "none":
        # Preserve the original 52.1 baseline reader and current static feature list behaviour.
        path_obj = Path(data_path)
        raw = get_df(data_path) if path_obj.is_dir() else read_training_frame(data_path)
        work, feature_cols, cat_cols = prepare_frame(
            raw,
            cfg.target_col,
            cfg.month_col,
            cfg.ignore_cols,
            cfg.add_time_features,
        )
        return {
            "work": work,
            "feature_cols": feature_cols,
            "cat_cols": cat_cols,
            "feature_selection_report": None,
            "feature_selection_metadata": None,
            "selection_paths": {},
        }

    dynamic_ignore = _dynamic_ignore_cols(data_path, cfg.target_col, cfg.ignore_cols)
    print(
        f"dynamic feature selection: method={fs_cfg.method}, "
        f"sample_rows={fs_cfg.sample_rows:,}, psi_threshold={fs_cfg.psi_threshold}"
    )
    print(f"dynamic ignore cols: {len(dynamic_ignore)}")

    # Feature selection uses only a bounded sample of the full-feature candidate source.
    sample_raw = sample_training_frame(
        data_path,
        max_rows=fs_cfg.sample_rows,
        random_state=fs_cfg.sample_random_state,
    )
    sample_df, candidate_features, sample_cat_cols = prepare_frame(
        sample_raw,
        target_col=cfg.target_col,
        month_col=cfg.month_col,
        ignore_cols=dynamic_ignore,
        add_time_features_flag=cfg.add_time_features,
        fill_numeric_missing=False,
    )
    fs_train, fs_valid, _ = split_frame(
        sample_df,
        month_col=cfg.month_col,
        train_end_exclusive=cfg.train_end_exclusive,
        valid_month=cfg.valid_month,
        test_month=cfg.test_month,
    )
    if train_start is not None:
        fs_train = fs_train.loc[fs_train[cfg.month_col] >= pd.Timestamp(train_start)].copy()

    # Loyalty: feature selection may use valid_model but never valid_calib.
    if selection_valid_clients is not None and "cdi_id" in fs_valid.columns:
        client_set = set(selection_valid_clients)
        fs_valid = fs_valid.loc[fs_valid["cdi_id"].isin(client_set)].copy()

    if fs_train.empty or fs_valid.empty:
        raise ValueError(
            "Dynamic FS sample does not contain train/selection-valid rows. "
            "Increase feature_selection.sample_rows or check split months."
        )

    selected_features, selection_report, selection_metadata = select_features(
        train=fs_train,
        valid=fs_valid,
        target_col=cfg.target_col,
        task_type=cfg.task_type,
        candidate_features=candidate_features,
        cat_cols=sample_cat_cols,
        month_col=cfg.month_col,
        config=fs_cfg,
    )
    selection_metadata.update({
        "model_key": cfg.model_key,
        "selection_train_rows": int(len(fs_train)),
        "selection_valid_rows": int(len(fs_valid)),
        "selection_valid_role": "valid_model" if selection_valid_clients is not None else "valid",
        "selection_train_months": [
            str(pd.Timestamp(x).date()) for x in sorted(fs_train[cfg.month_col].dropna().unique())
        ],
        "selection_valid_months": [
            str(pd.Timestamp(x).date()) for x in sorted(fs_valid[cfg.month_col].dropna().unique())
        ],
        "dynamic_ignore_cols": dynamic_ignore,
    })
    selection_paths = save_selection_report(
        selection_report,
        selection_metadata,
        model_key=cfg.model_key,
        config=fs_cfg,
    )
    print(f"selected features: {len(selected_features)} / {len(candidate_features)}")

    # Read the full candidate table only after selection and only with required source columns.
    source_columns = _source_columns_for_selected(
        data_path,
        selected_features=selected_features,
        target_col=cfg.target_col,
        month_col=cfg.month_col,
        ignore_cols=dynamic_ignore,
        add_time_features_flag=cfg.add_time_features,
    )
    raw = read_training_frame(data_path, columns=source_columns)
    work, _, prepared_cat_cols = prepare_frame(
        raw,
        target_col=cfg.target_col,
        month_col=cfg.month_col,
        ignore_cols=dynamic_ignore,
        add_time_features_flag=cfg.add_time_features,
        fill_numeric_missing=False,
    )
    feature_cols = [c for c in selected_features if c in work.columns]
    cat_cols = [c for c in prepared_cat_cols if c in feature_cols]
    missing = [c for c in selected_features if c not in work.columns]
    if missing:
        print("WARNING selected features absent after full read:", missing[:20])
    if not feature_cols:
        raise ValueError("No selected features left after full-data read")

    return {
        "work": work,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "feature_selection_report": selection_report,
        "feature_selection_metadata": selection_metadata,
        "selection_paths": selection_paths,
    }


def _derive_loyalty_valid_clients(
    data_path: str,
    valid_month: str,
    month_col: str = "forecast_month",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the same client-level validation split before dynamic FS.

    Only cdi_id + month are read, so the split can be fixed without loading all features.
    """
    ids = read_training_frame(data_path, columns=["cdi_id", month_col])
    ids[month_col] = pd.to_datetime(ids[month_col]).dt.to_period("M").dt.to_timestamp()
    month = pd.Timestamp(valid_month).to_period("M").to_timestamp()
    valid_clients = ids.loc[ids[month_col] == month, "cdi_id"].dropna().unique()
    if len(valid_clients) < 2:
        raise ValueError("Not enough clients in valid to split valid_model / valid_calib")
    return train_test_split(valid_clients, test_size=0.5, random_state=random_state)


def train_cashback_connected_clean(
    data_path="data/cb_train_feat_select_connected_25_05_cashback_total_pred",
    train_start="2025-09-01",
    train_end_exclusive="2026-05-01",
    valid_month="2026-05-01",
    test_month="2026-06-01",
    model_dir="models",
    feature_selection: Optional[dict] = None,
    model_key: str = "cashback_connected",
    model_filename: str = "cashback_connected_feat_sel_catboost.cbm",
    output_split_dir: str = "data/files/train_pipeline",
    iterations: int = 2000,
    learning_rate: float = 0.03,
    depth: int = 6,
    l2_leaf_reg: float = 10.0,
    early_stopping_rounds: int = 100,
):
    """Train cashback regression with optional dynamic feature selection."""
    t0 = time.time()

    cfg = TrainJobConfig(
        model_key=model_key,
        task_type="regression",
        input_path=data_path,
        target_col="target_cashback_month",
        output_split_dir=output_split_dir,
        model_output_dir=model_dir,
        model_filename=model_filename,
        ignore_cols=[
            "cdi_id",
            "target_pos_turnover_month",
            "target_pos_txn_cnt_month",
            "target_cashback_positive_flg",
            "target_loyalty_connected_flg",
            "tel_sms",
            "mobile_phone",
        ],
        train_end_exclusive=train_end_exclusive,
        valid_month=valid_month,
        test_month=test_month,
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        early_stopping_rounds=early_stopping_rounds,
        feature_selection=feature_selection,
    )

    prepared = _prepare_static_or_dynamic(data_path=data_path, cfg=cfg, train_start=train_start)
    work = prepared["work"]
    feature_cols = prepared["feature_cols"]
    cat_cols = prepared["cat_cols"]
    work["forecast_month"] = pd.to_datetime(work["forecast_month"])

    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(cfg.train_end_exclusive)
    valid_month_ts = pd.Timestamp(cfg.valid_month)

    train = work.loc[
        (work["forecast_month"] >= train_start_ts)
        & (work["forecast_month"] < train_end_ts)
    ].copy()
    valid = work.loc[work["forecast_month"] == valid_month_ts].copy()
    if cfg.test_month is not None:
        test = work.loc[work["forecast_month"] == pd.Timestamp(cfg.test_month)].copy()
    else:
        test = pd.DataFrame(columns=work.columns)

    print("train shape:", train.shape)
    print("valid shape:", valid.shape)
    print("test shape:", test.shape)
    if train.empty:
        raise ValueError("train is empty")
    if valid.empty:
        raise ValueError(f"valid is empty for valid_month={cfg.valid_month}")
    if cfg.test_month is not None and test.empty:
        raise ValueError(
            f"OOT/test is empty for test_month={cfg.test_month}. "
            "Independent OOT is required for current-vs-dynamic comparison."
        )

    cat_idx = [feature_cols.index(c) for c in cat_cols if c in feature_cols]
    train_pool = Pool(train[feature_cols], train[cfg.target_col], cat_features=cat_idx)
    valid_pool = Pool(valid[feature_cols], valid[cfg.target_col], cat_features=cat_idx)

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=cfg.iterations,
        learning_rate=cfg.learning_rate,
        depth=cfg.depth,
        l2_leaf_reg=cfg.l2_leaf_reg,
        random_seed=42,
        od_type="Iter",
        od_wait=cfg.early_stopping_rounds,
        verbose=100,
        allow_writing_files=False,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    # Preserve the original business post-processing exactly.
    pred_valid = np.clip(model.predict(valid[feature_cols]), 0, 5000)
    metrics = {"valid": regression_metrics_dict(valid[cfg.target_col], pred_valid)}

    pred_test = None
    if not test.empty:
        pred_test = np.clip(model.predict(test[feature_cols]), 0, 5000)
        metrics["test"] = regression_metrics_dict(test[cfg.target_col], pred_test)
    else:
        print("test absent — valid metrics only")

    paths = save_model_artifacts(model, cfg, feature_cols, cat_cols, metrics)
    paths.update(prepared["selection_paths"])
    elapsed = float(time.time() - t0)

    del train_pool, valid_pool
    gc.collect()
    return {
        "config": cfg,
        "model": model,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "train": train,
        "valid": valid,
        "test": test,
        "pred_valid": pred_valid,
        "pred_test": pred_test,
        "metrics": metrics,
        "paths": paths,
        "feature_selection_report": prepared["feature_selection_report"],
        "feature_selection_metadata": prepared["feature_selection_metadata"],
        "elapsed_seconds": elapsed,
    }


def train_loyalty_connection_clean(
    data_path="data/cb_sample_for_binary_25_05_cashback_total_pred",
    train_start="2025-09-01",
    train_end_exclusive="2026-05-01",
    valid_month="2026-05-01",
    test_month="2026-06-01",
    model_dir="models",
    feature_selection: Optional[dict] = None,
    model_key: str = "loyalty_connect",
    model_filename: str = "loyalty_connect_for_catboost.cbm",
    output_split_dir: str = "data/files/train_pipeline",
    fixed_valid_model_clients: Optional[Iterable] = None,
    calibrator_filename: Optional[str] = None,
    iterations: int = 2000,
    learning_rate: float = 0.03,
    depth: int = 6,
    l2_leaf_reg: float = 10.0,
    early_stopping_rounds: int = 100,
):
    """Train loyalty classifier with optional dynamic FS and untouched calibration half."""
    t0 = time.time()

    cfg = TrainJobConfig(
        model_key=model_key,
        task_type="classification",
        input_path=data_path,
        target_col="target_loyalty_connected_flg",
        output_split_dir=output_split_dir,
        model_output_dir=model_dir,
        model_filename=model_filename,
        ignore_cols=[
            "cdi_id",
            "period",
            "tel_sms",
            "mobile_phone",
            "target_selected_categories_cnt",
            "target_cashback_month",
            "target_pos_turnover_month",
            "target_pos_txn_cnt_month",
            "target_cashback_positive_flg",
        ],
        train_end_exclusive=train_end_exclusive,
        valid_month=valid_month,
        test_month=test_month,
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        early_stopping_rounds=early_stopping_rounds,
        auto_class_weights="Balanced",
        calibrate_isotonic=True,
        feature_selection=feature_selection,
    )

    # For dynamic FS determine the client holdout before looking at features.
    fs_cfg = normalize_feature_selection_config(feature_selection)
    if fixed_valid_model_clients is not None:
        clients_model = np.asarray(list(fixed_valid_model_clients))
        clients_calib = None
    elif fs_cfg.method != "none":
        clients_model, clients_calib = _derive_loyalty_valid_clients(
            data_path=data_path,
            valid_month=valid_month,
            month_col=cfg.month_col,
            random_state=42,
        )
    else:
        clients_model = None
        clients_calib = None

    prepared = _prepare_static_or_dynamic(
        data_path=data_path,
        cfg=cfg,
        train_start=train_start,
        selection_valid_clients=clients_model if fs_cfg.method != "none" else None,
    )
    work = prepared["work"]
    feature_cols = prepared["feature_cols"]
    cat_cols = prepared["cat_cols"]
    work["forecast_month"] = pd.to_datetime(work["forecast_month"])
    work[cfg.target_col] = work[cfg.target_col].astype(int)

    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(cfg.train_end_exclusive)
    valid_month_ts = pd.Timestamp(cfg.valid_month)

    train = work.loc[
        (work["forecast_month"] >= train_start_ts)
        & (work["forecast_month"] < train_end_ts)
    ].copy()
    valid_full = work.loc[work["forecast_month"] == valid_month_ts].copy()
    if cfg.test_month is not None:
        test = work.loc[work["forecast_month"] == pd.Timestamp(cfg.test_month)].copy()
    else:
        test = pd.DataFrame(columns=work.columns)

    if train.empty:
        raise ValueError("train is empty")
    if valid_full.empty:
        raise ValueError(f"valid is empty for valid_month={cfg.valid_month}")
    if cfg.test_month is not None and test.empty:
        raise ValueError(
            f"OOT/test is empty for test_month={cfg.test_month}. "
            "Independent OOT is required for current-vs-dynamic comparison."
        )

    # Static baseline: preserve original split. Benchmark can pass the current model clients into
    # every dynamic strategy so all strategies use exactly the same client holdout.
    if clients_model is None:
        valid_clients = valid_full["cdi_id"].dropna().unique()
        if len(valid_clients) < 2:
            raise ValueError("Not enough clients in valid for valid_model / valid_calib")
        clients_model, clients_calib = train_test_split(
            valid_clients,
            test_size=0.5,
            random_state=42,
        )
    else:
        client_set = set(clients_model)
        all_valid_clients = valid_full["cdi_id"].dropna().unique()
        clients_calib = np.asarray([x for x in all_valid_clients if x not in client_set])

    valid_model = valid_full.loc[valid_full["cdi_id"].isin(set(clients_model))].copy()
    valid_calib = valid_full.loc[valid_full["cdi_id"].isin(set(clients_calib))].copy()
    if valid_model.empty or valid_calib.empty:
        raise ValueError("valid_model or valid_calib is empty")

    print("train shape:", train.shape)
    print("valid_full shape:", valid_full.shape)
    print("valid_model shape:", valid_model.shape)
    print("valid_calib shape:", valid_calib.shape)
    print("test shape:", test.shape)

    cat_idx = [feature_cols.index(c) for c in cat_cols if c in feature_cols]
    train_pool = Pool(train[feature_cols], train[cfg.target_col], cat_features=cat_idx)
    valid_pool = Pool(valid_model[feature_cols], valid_model[cfg.target_col], cat_features=cat_idx)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=cfg.iterations,
        learning_rate=cfg.learning_rate,
        depth=cfg.depth,
        l2_leaf_reg=cfg.l2_leaf_reg,
        random_seed=42,
        od_type="Iter",
        od_wait=cfg.early_stopping_rounds,
        auto_class_weights=cfg.auto_class_weights,
        verbose=100,
        allow_writing_files=False,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    pred_valid_model = model.predict_proba(valid_model[feature_cols])[:, 1]
    pred_valid_calib_raw = model.predict_proba(valid_calib[feature_cols])[:, 1]

    calibrator = None
    pred_valid_calib = pred_valid_calib_raw
    if cfg.calibrate_isotonic:
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(pred_valid_calib_raw, valid_calib[cfg.target_col])
        pred_valid_calib = calibrator.transform(pred_valid_calib_raw)

    metrics = {
        "valid_model_raw": classification_metrics_dict(valid_model[cfg.target_col], pred_valid_model),
        "valid_calib_raw": classification_metrics_dict(valid_calib[cfg.target_col], pred_valid_calib_raw),
        "valid_calib": classification_metrics_dict(valid_calib[cfg.target_col], pred_valid_calib),
    }

    pred_test_raw = None
    pred_test = None
    if not test.empty:
        pred_test_raw = model.predict_proba(test[feature_cols])[:, 1]
        pred_test = calibrator.transform(pred_test_raw) if calibrator is not None else pred_test_raw
        metrics["test_raw"] = classification_metrics_dict(test[cfg.target_col], pred_test_raw)
        metrics["test"] = classification_metrics_dict(test[cfg.target_col], pred_test)
    else:
        print("test absent — valid metrics only")

    paths = save_model_artifacts(model, cfg, feature_cols, cat_cols, metrics, calibrator=calibrator)
    paths.update(prepared["selection_paths"])

    if calibrator is not None:
        cal_name = calibrator_filename or (
            "isotonic_calibrator_loyalty.pkl"
            if model_key == "loyalty_connect"
            else f"isotonic_calibrator_{model_key}.pkl"
        )
        calibrator_path = Path(model_dir) / cal_name
        calibrator_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(calibrator, calibrator_path)
        paths["calibrator_path"] = str(calibrator_path)
        print(f"calibrator saved: {calibrator_path}")

    elapsed = float(time.time() - t0)
    del train_pool, valid_pool
    gc.collect()

    return {
        "config": cfg,
        "model": model,
        "calibrator": calibrator,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "train": train,
        "valid": valid_full,
        "valid_model": valid_model,
        "valid_calib": valid_calib,
        "test": test,
        "pred_valid_model": pred_valid_model,
        "pred_valid_calib_raw": pred_valid_calib_raw,
        "pred_valid_calib": pred_valid_calib,
        "pred_test_raw": pred_test_raw,
        "pred_test": pred_test,
        "metrics": metrics,
        "paths": paths,
        "feature_selection_report": prepared["feature_selection_report"],
        "feature_selection_metadata": prepared["feature_selection_metadata"],
        "elapsed_seconds": elapsed,
    }
