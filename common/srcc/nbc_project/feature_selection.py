"""
Dynamic feature selection for NBC models.

Supported strategies:
- none: keep all candidate features (current behaviour for already selected *_fs_sel inputs)
- psi: remove unstable features by Population Stability Index
- importance: quick CatBoost model -> top-N by PredictionValuesChange importance
- rfe: CatBoost native recursive feature elimination
- permutation: validation permutation importance -> top-N (cheap baseline)
- metric_rfe: CatBoost RFE ranking -> dynamic valid-metric cutoff / bootstrap 1-SE
- permutation_ci: permutation importance -> CI-driven cutoff, no final top-N
- null_importance: real CatBoost importance vs shuffled-target noise distribution
- stability: rolling temporal stability selection
- psi_importance: PSI stability filter -> CatBoost top-N
- psi_rfe: PSI stability filter -> CatBoost RFE
- psi_metric_rfe: PSI -> metric-guided RFE cutoff
- psi_permutation: PSI -> validation permutation importance
- psi_permutation_ci: PSI -> permutation CI cutoff
- psi_null_importance: PSI -> null-importance significance cutoff
- psi_stability: PSI -> rolling temporal stability selection

Important design choice: selection uses train + validation only. Test/OOT is never used for
feature selection, so it remains a fair comparison set against the current implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Iterable, Optional
import json

import numpy as np
import pandas as pd
from catboost import (
    Pool,
    CatBoostRegressor,
    CatBoostClassifier,
    EFeaturesSelectionAlgorithm,
    EShapCalcType,
)


SUPPORTED_METHODS = {
    "none",
    "psi",
    "importance",
    "rfe",
    "metric_rfe",
    "permutation",
    "permutation_ci",
    "null_importance",
    "stability",
    "psi_importance",
    "psi_rfe",
    "psi_metric_rfe",
    "psi_permutation",
    "psi_permutation_ci",
    "psi_null_importance",
    "psi_stability",
}


@dataclass
class FeatureSelectionConfig:
    method: str = "none"

    # PSI
    psi_threshold: float = 0.20
    psi_bins: int = 10
    psi_reference_months: int = 3
    psi_max_categories: int = 50
    # Calendar/time features have deterministic month-to-month drift by design.
    # PSI should not reject them, but importance/RFE may still remove them.
    psi_exempt_features: Optional[list[str]] = None

    # Cheap prefilter before PSI/model selection
    max_missing_share: float = 0.995
    min_unique_values: int = 2

    # Correlation prefilter (train only, numeric features only).
    # For a highly-correlated pair we keep the more complete feature
    # (lower missing share; then higher cardinality as a deterministic tie-breaker).
    drop_correlated_features: bool = True
    correlation_threshold: float = 0.95
    correlation_method: str = "pearson"  # pearson / spearman
    correlation_max_rows: int = 20_000
    # Low-cardinality integer features may actually be categorical codes; do not correlation-drop them.
    correlation_min_unique_values: int = 20
    correlation_random_state: int = 42
    correlation_exempt_features: Optional[list[str]] = None

    # Model-based selection
    top_n: int = 200
    selector_iterations: int = 300
    selector_learning_rate: float = 0.05
    selector_depth: int = 6
    selector_l2_leaf_reg: float = 5.0
    selector_early_stopping_rounds: int = 50
    selector_random_seed: int = 42
    selector_auto_class_weights: Optional[str] = None

    # Dynamic top-N. If enabled, N is chosen ONLY by selection valid; OOT/test stays untouched.
    auto_top_n: bool = False
    top_n_candidates: Optional[list[int]] = None
    # Prefer the smallest N whose valid metric is within this tolerance of the best one.
    # Regression: RMSE <= best + tolerance. Classification: AUC >= best - tolerance.
    top_n_tolerance: float = 0.0

    # RFE
    rfe_steps: int = 5
    rfe_algorithm: str = "prediction_values_change"  # shap_values / loss_function_change
    rfe_shap_calc_type: str = "regular"  # approximate / regular / exact

    # Metric-guided RFE. The final feature count is NOT a predefined top-N.
    # RFE creates a nested elimination path; valid quality chooses the cutoff.
    metric_rfe_min_features: int = 30
    metric_rfe_drop_fraction: float = 0.15
    metric_rfe_min_drop: int = 10
    metric_rfe_max_models: int = 20
    metric_rfe_rule: str = "one_se"  # best / tolerance / one_se
    metric_rfe_tolerance: float = 0.0
    metric_rfe_bootstrap_repeats: int = 100
    metric_rfe_one_se_multiplier: float = 1.0
    metric_rfe_eval_rows: int = 50_000

    # Permutation importance on selection-valid.
    permutation_repeats: int = 3
    permutation_eval_rows: int = 50_000
    permutation_max_features: Optional[int] = 400

    # CI-driven permutation cutoff. Final N is determined by evidence that permutation
    # genuinely worsens validation quality, not by rank position.
    permutation_ci_bootstrap_repeats: int = 40
    permutation_ci_bootstrap_rows: int = 5_000
    permutation_ci_confidence: float = 0.95
    permutation_ci_min_importance: float = 0.0
    # Optional compute-only prescreen. None keeps the method fully free of N-based cutoffs.
    permutation_ci_max_features: Optional[int] = None
    # Optional emergency fallback. None = fail loudly when no feature is significant.
    permutation_ci_fallback_features: Optional[int] = None

    # Null importance: compare real importance with importance under shuffled target.
    null_importance_repeats: int = 20
    null_importance_quantile: float = 0.95
    null_importance_max_features: Optional[int] = None
    null_importance_fallback_features: Optional[int] = None

    # Temporal stability selection. Each fold trains only on earlier months.
    stability_base_method: str = "importance"  # importance / permutation / rfe_loss / rfe_shap / rfe_pvc
    stability_n_splits: int = 4
    stability_min_train_months: int = 3
    stability_selection_rate: float = 0.75
    stability_top_n_per_fold: int = 200
    stability_apply_psi_per_fold: bool = True
    # Optional safety cap only. None means selection-rate alone determines final N.
    stability_max_features: Optional[int] = None

    # Sampling is consumed by training_workflow when source is a large full parquet dataset.
    sample_rows: int = 300_000
    sample_random_state: int = 42

    # Features listed here survive all filters. Useful for business-critical features.
    protected_features: Optional[list[str]] = None

    # Artifacts
    report_dir: str = "reports/feature_selection"


def normalize_feature_selection_config(config: Optional[dict | FeatureSelectionConfig]) -> FeatureSelectionConfig:
    if config is None:
        return FeatureSelectionConfig()
    if isinstance(config, FeatureSelectionConfig):
        result = config
    elif isinstance(config, dict):
        result = FeatureSelectionConfig(**config)
    else:
        raise TypeError("feature_selection must be dict, FeatureSelectionConfig or None")

    result.method = str(result.method).lower().strip()
    if result.method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown feature selection method={result.method!r}. Supported: {sorted(SUPPORTED_METHODS)}")
    if not (0 < float(result.correlation_threshold) <= 1):
        raise ValueError("correlation_threshold must be in (0, 1]")
    result.correlation_method = str(result.correlation_method).lower().strip()
    if result.correlation_method not in {"pearson", "spearman"}:
        raise ValueError("correlation_method must be 'pearson' or 'spearman'")
    if int(result.correlation_max_rows) < 100:
        raise ValueError("correlation_max_rows must be >= 100")
    if int(result.correlation_min_unique_values) < 2:
        raise ValueError("correlation_min_unique_values must be >= 2")
    if result.top_n <= 0:
        raise ValueError("top_n must be > 0")
    if result.psi_bins < 2:
        raise ValueError("psi_bins must be >= 2")
    if result.psi_reference_months < 1:
        raise ValueError("psi_reference_months must be >= 1")
    if result.top_n_candidates is not None:
        result.top_n_candidates = sorted({int(x) for x in result.top_n_candidates if int(x) > 0})
        if not result.top_n_candidates:
            raise ValueError("top_n_candidates must contain at least one positive integer")
    if result.top_n_tolerance < 0:
        raise ValueError("top_n_tolerance must be >= 0")
    if result.permutation_repeats < 1:
        raise ValueError("permutation_repeats must be >= 1")
    if result.permutation_eval_rows < 1:
        raise ValueError("permutation_eval_rows must be >= 1")
    if result.metric_rfe_min_features < 1:
        raise ValueError("metric_rfe_min_features must be >= 1")
    if not (0 < result.metric_rfe_drop_fraction < 1):
        raise ValueError("metric_rfe_drop_fraction must be in (0, 1)")
    if result.metric_rfe_min_drop < 1:
        raise ValueError("metric_rfe_min_drop must be >= 1")
    if result.metric_rfe_max_models < 2:
        raise ValueError("metric_rfe_max_models must be >= 2")
    if str(result.metric_rfe_rule).lower().strip() not in {"best", "tolerance", "one_se"}:
        raise ValueError("metric_rfe_rule must be one of: best, tolerance, one_se")
    result.metric_rfe_rule = str(result.metric_rfe_rule).lower().strip()
    if result.metric_rfe_tolerance < 0:
        raise ValueError("metric_rfe_tolerance must be >= 0")
    if result.metric_rfe_bootstrap_repeats < 2:
        raise ValueError("metric_rfe_bootstrap_repeats must be >= 2")
    if result.metric_rfe_one_se_multiplier < 0:
        raise ValueError("metric_rfe_one_se_multiplier must be >= 0")
    if result.metric_rfe_eval_rows < 1:
        raise ValueError("metric_rfe_eval_rows must be >= 1")
    if result.permutation_ci_bootstrap_repeats < 2:
        raise ValueError("permutation_ci_bootstrap_repeats must be >= 2")
    if result.permutation_ci_bootstrap_rows < 10:
        raise ValueError("permutation_ci_bootstrap_rows must be >= 10")
    if not (0 < result.permutation_ci_confidence < 1):
        raise ValueError("permutation_ci_confidence must be in (0, 1)")
    if result.permutation_ci_max_features is not None and result.permutation_ci_max_features < 1:
        raise ValueError("permutation_ci_max_features must be >= 1 or None")
    if result.permutation_ci_fallback_features is not None and result.permutation_ci_fallback_features < 1:
        raise ValueError("permutation_ci_fallback_features must be >= 1 or None")
    if result.null_importance_repeats < 2:
        raise ValueError("null_importance_repeats must be >= 2")
    if not (0 < result.null_importance_quantile < 1):
        raise ValueError("null_importance_quantile must be in (0, 1)")
    if result.null_importance_max_features is not None and result.null_importance_max_features < 1:
        raise ValueError("null_importance_max_features must be >= 1 or None")
    if result.null_importance_fallback_features is not None and result.null_importance_fallback_features < 1:
        raise ValueError("null_importance_fallback_features must be >= 1 or None")
    if result.stability_max_features is not None and result.stability_max_features < 1:
        raise ValueError("stability_max_features must be >= 1 or None")
    if not (0 < result.stability_selection_rate <= 1):
        raise ValueError("stability_selection_rate must be in (0, 1]")
    if result.stability_n_splits < 1:
        raise ValueError("stability_n_splits must be >= 1")
    if result.stability_min_train_months < 1:
        raise ValueError("stability_min_train_months must be >= 1")
    if result.psi_exempt_features is None:
        result.psi_exempt_features = [
            "forecast_month_num",
            "forecast_quarter",
            "forecast_year",
            "season_total_pos_coef_honest",
        ]
    return result


def _unique_keep_order(items: Iterable[str]) -> list[str]:
    out, seen = [], set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _safe_distribution(counts: pd.Series, categories: list, eps: float = 1e-6) -> np.ndarray:
    values = counts.reindex(categories, fill_value=0).astype(float).to_numpy()
    total = values.sum()
    if total <= 0:
        values = np.full(len(categories), 1.0 / max(len(categories), 1))
    else:
        values = values / total
    values = np.clip(values, eps, None)
    values = values / values.sum()
    return values


def _numeric_buckets(reference: pd.Series, current: pd.Series, bins: int) -> tuple[pd.Series, pd.Series]:
    ref_num = pd.to_numeric(reference, errors="coerce")
    cur_num = pd.to_numeric(current, errors="coerce")
    non_null = ref_num.dropna()

    if non_null.nunique() <= 1:
        # Constant in reference: PSI is still meaningful through missing/non-missing split.
        ref_bucket = pd.Series(np.where(ref_num.isna(), "__MISSING__", "__VALUE__"), index=reference.index)
        cur_bucket = pd.Series(np.where(cur_num.isna(), "__MISSING__", "__VALUE__"), index=current.index)
        return ref_bucket, cur_bucket

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(non_null.quantile(quantiles).to_numpy(dtype=float))

    if len(edges) < 3:
        # Too few unique numeric cut points -> treat unique values as categories.
        ref_bucket = ref_num.astype("object").where(ref_num.notna(), "__MISSING__").astype(str)
        cur_bucket = cur_num.astype("object").where(cur_num.notna(), "__MISSING__").astype(str)
        return ref_bucket, cur_bucket

    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_bucket = pd.cut(ref_num, bins=edges, include_lowest=True, duplicates="drop").astype("object")
    cur_bucket = pd.cut(cur_num, bins=edges, include_lowest=True, duplicates="drop").astype("object")
    ref_bucket = ref_bucket.where(ref_num.notna(), "__MISSING__").astype(str)
    cur_bucket = cur_bucket.where(cur_num.notna(), "__MISSING__").astype(str)
    return ref_bucket, cur_bucket


def _categorical_buckets(reference: pd.Series, current: pd.Series, max_categories: int) -> tuple[pd.Series, pd.Series]:
    ref = reference.astype("object").where(reference.notna(), "__MISSING__").astype(str)
    cur = current.astype("object").where(current.notna(), "__MISSING__").astype(str)

    top = ref.value_counts(dropna=False).head(max_categories).index.astype(str).tolist()
    allowed = set(top)
    allowed.add("__MISSING__")

    ref = ref.where(ref.isin(allowed), "__OTHER__")
    cur = cur.where(cur.isin(allowed), "__OTHER__")
    return ref, cur


def calculate_psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
    max_categories: int = 50,
) -> float:
    """Population Stability Index. Bins/categories are defined only from reference."""
    is_numeric = pd.api.types.is_numeric_dtype(reference) and not pd.api.types.is_bool_dtype(reference)

    if is_numeric:
        ref_bucket, cur_bucket = _numeric_buckets(reference, current, bins=bins)
    else:
        ref_bucket, cur_bucket = _categorical_buckets(reference, current, max_categories=max_categories)

    categories = sorted(set(ref_bucket.astype(str).unique()) | set(cur_bucket.astype(str).unique()))
    if not categories:
        return 0.0

    ref_dist = _safe_distribution(ref_bucket.value_counts(), categories)
    cur_dist = _safe_distribution(cur_bucket.value_counts(), categories)
    psi = np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist))
    return float(psi)


def _recent_reference(train: pd.DataFrame, month_col: str, n_months: int) -> pd.DataFrame:
    if month_col not in train.columns:
        return train
    months = sorted(pd.to_datetime(train[month_col].dropna()).unique())
    if not months:
        return train
    keep = set(months[-n_months:])
    return train[pd.to_datetime(train[month_col]).isin(keep)]


def compute_psi_table(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: list[str],
    month_col: str,
    config: FeatureSelectionConfig,
) -> pd.DataFrame:
    """PSI between recent train history and selection validation month."""
    reference = _recent_reference(train, month_col, config.psi_reference_months)
    rows = []
    for col in feature_cols:
        try:
            psi = calculate_psi(
                reference[col],
                valid[col],
                bins=config.psi_bins,
                max_categories=config.psi_max_categories,
            )
        except Exception as exc:
            psi = np.nan
            error = str(exc)
        else:
            error = None
        rows.append({
            "feature": col,
            "psi": psi,
            "psi_threshold": config.psi_threshold,
            "psi_pass": bool(pd.notna(psi) and psi <= config.psi_threshold),
            "psi_error": error,
            "psi_reference_rows": int(len(reference)),
            "psi_current_rows": int(len(valid)),
        })
    return pd.DataFrame(rows).sort_values(["psi_pass", "psi"], ascending=[True, False], na_position="first")


def prefilter_features(
    train: pd.DataFrame,
    feature_cols: list[str],
    config: FeatureSelectionConfig,
    cat_cols: Optional[list[str]] = None,
) -> tuple[list[str], pd.DataFrame]:
    """Cheap train-only prefilter: missing/constant -> high correlation.

    Correlation filtering is intentionally target-independent and uses only numeric
    features from TRAIN. Categorical CatBoost features are never removed by this step.
    For each highly-correlated pair we prefer protected/exempt features, then the
    feature with lower missing share and, as a tie-breaker, higher cardinality.
    """
    protected = set(config.protected_features or [])
    corr_exempt = set(config.correlation_exempt_features or [])
    cat_set = set(cat_cols or [])

    kept_basic: list[str] = []
    rows: list[dict] = []
    stats: dict[str, tuple[float, int]] = {}

    for col in feature_cols:
        s = train[col]
        missing_share = float(s.isna().mean())
        nunique = int(s.nunique(dropna=True))
        stats[col] = (missing_share, nunique)
        reason = None
        if col not in protected:
            if missing_share > config.max_missing_share:
                reason = "too_many_missing"
            elif nunique < config.min_unique_values:
                reason = "constant_or_single_value"
        selected = reason is None
        if selected:
            kept_basic.append(col)
        rows.append({
            "feature": col,
            "missing_share_train": missing_share,
            "nunique_train": nunique,
            "prefilter_pass": selected,
            "prefilter_reason": reason,
            "protected": col in protected,
            "correlation_exempt": col in corr_exempt,
            "correlation_checked": False,
            "correlation_with": None,
            "correlation_abs": np.nan,
        })

    report = pd.DataFrame(rows)
    if not kept_basic or not config.drop_correlated_features:
        return kept_basic, report

    # Only numeric non-categorical columns participate in the correlation matrix.
    numeric_cols = [
        c for c in kept_basic
        if c not in cat_set
        and pd.api.types.is_numeric_dtype(train[c])
        and stats[c][1] >= int(config.correlation_min_unique_values)
    ]
    if len(numeric_cols) < 2:
        return kept_basic, report

    corr_source = train[numeric_cols]
    max_rows = int(config.correlation_max_rows)
    if len(corr_source) > max_rows:
        corr_source = corr_source.sample(
            n=max_rows,
            random_state=int(config.correlation_random_state),
        )

    # float32 materially reduces the temporary memory footprint for wide feature stores.
    corr_source = corr_source.astype("float32", copy=False)
    corr = corr_source.corr(method=config.correlation_method).abs()

    # Protected/exempt features get first priority. For ordinary features prefer lower
    # missingness, then greater cardinality. Final feature-name sort keeps it deterministic.
    def _priority(col: str):
        missing_share, nunique = stats[col]
        forced = col in protected or col in corr_exempt
        return (0 if forced else 1, missing_share, -nunique, col)

    ordered = sorted(numeric_cols, key=_priority)
    kept_numeric: list[str] = []
    dropped_map: dict[str, tuple[str, float]] = {}

    for col in ordered:
        is_forced = col in protected or col in corr_exempt
        conflicts: list[tuple[str, float]] = []
        for prev in kept_numeric:
            value = corr.at[col, prev]
            if pd.notna(value) and float(value) >= float(config.correlation_threshold):
                conflicts.append((prev, float(value)))

        if conflicts and not is_forced:
            keeper, value = max(conflicts, key=lambda x: x[1])
            dropped_map[col] = (keeper, value)
        else:
            kept_numeric.append(col)

    if dropped_map:
        idx = report.set_index("feature").index
        for col, (keeper, value) in dropped_map.items():
            mask = report["feature"].eq(col)
            report.loc[mask, "prefilter_pass"] = False
            report.loc[mask, "prefilter_reason"] = "high_correlation"
            report.loc[mask, "correlation_with"] = keeper
            report.loc[mask, "correlation_abs"] = value

    checked_mask = report["feature"].isin(numeric_cols)
    report.loc[checked_mask, "correlation_checked"] = True

    kept = [c for c in kept_basic if c not in dropped_map]
    report.attrs["correlation_threshold"] = float(config.correlation_threshold)
    report.attrs["correlation_method"] = config.correlation_method
    report.attrs["correlation_rows"] = int(len(corr_source))
    report.attrs["correlation_dropped"] = int(len(dropped_map))
    return kept, report


def _selector_model(task_type: str, config: FeatureSelectionConfig):
    common = dict(
        iterations=config.selector_iterations,
        learning_rate=config.selector_learning_rate,
        depth=config.selector_depth,
        l2_leaf_reg=config.selector_l2_leaf_reg,
        random_seed=config.selector_random_seed,
        early_stopping_rounds=config.selector_early_stopping_rounds,
        verbose=False,
        allow_writing_files=False,
    )
    if task_type == "classification":
        if config.selector_auto_class_weights:
            common["auto_class_weights"] = config.selector_auto_class_weights
        return CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", **common)
    return CatBoostRegressor(loss_function="RMSE", eval_metric="RMSE", **common)


def _make_pool(df: pd.DataFrame, target_col: str, feature_cols: list[str], cat_cols: list[str]) -> Pool:
    cat_idx = [feature_cols.index(c) for c in cat_cols if c in feature_cols]
    return Pool(df[feature_cols], df[target_col], cat_features=cat_idx, feature_names=feature_cols)



def _predict_selector_model(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    task_type: str,
) -> np.ndarray:
    cat_idx = [feature_cols.index(c) for c in cat_cols if c in feature_cols]
    pool = Pool(df[feature_cols], cat_features=cat_idx, feature_names=feature_cols)
    if task_type == "classification":
        pred = model.predict_proba(pool)
        pred = np.asarray(pred)
        return pred[:, 1] if pred.ndim == 2 else pred.reshape(-1)
    return np.asarray(model.predict(pool)).reshape(-1)


def _primary_metric(y_true: pd.Series, pred: np.ndarray, task_type: str) -> tuple[str, float]:
    from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss

    y = np.asarray(y_true)
    pred = np.asarray(pred)
    if task_type == "classification":
        if pd.Series(y).nunique(dropna=True) >= 2:
            return "roc_auc", float(roc_auc_score(y.astype(int), pred))
        # Rare safety fallback for a tiny sampled validation month.
        return "logloss", float(log_loss(y.astype(int), np.clip(pred, 1e-7, 1 - 1e-7), labels=[0, 1]))
    return "rmse", float(np.sqrt(mean_squared_error(y, pred)))


def _metric_higher_is_better(metric_name: str) -> bool:
    return metric_name in {"roc_auc", "accuracy", "f1", "average_precision"}


def _metric_from_name(y_true, pred, metric_name: str) -> float:
    from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss

    y = np.asarray(y_true)
    pred = np.asarray(pred)
    if metric_name == "rmse":
        return float(np.sqrt(mean_squared_error(y, pred)))
    if metric_name == "roc_auc":
        if pd.Series(y).nunique(dropna=True) < 2:
            return np.nan
        return float(roc_auc_score(y.astype(int), pred))
    if metric_name == "logloss":
        return float(log_loss(y.astype(int), np.clip(pred, 1e-7, 1 - 1e-7), labels=[0, 1]))
    raise ValueError(f"Unsupported metric_name={metric_name!r}")


def _bootstrap_metric_se(
    y_true,
    pred,
    metric_name: str,
    repeats: int,
    random_seed: int,
) -> tuple[float, list[float]]:
    """Bootstrap uncertainty of the validation metric using already computed predictions."""
    y = np.asarray(y_true)
    pred = np.asarray(pred)
    n = len(y)
    if n < 2:
        return 0.0, []
    rng = np.random.default_rng(random_seed)
    values = []
    for _ in range(int(repeats)):
        idx = rng.integers(0, n, size=n)
        value = _metric_from_name(y[idx], pred[idx], metric_name)
        if np.isfinite(value):
            values.append(float(value))
    if len(values) < 2:
        return 0.0, values
    return float(np.std(values, ddof=1)), values


def _quality_delta(baseline: float, changed: float, metric_name: str) -> float:
    """Positive value always means the changed prediction is worse."""
    if _metric_higher_is_better(metric_name):
        return float(baseline - changed)
    return float(changed - baseline)


def _fit_and_score_feature_set(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target_col: str,
    task_type: str,
    feature_cols: list[str],
    cat_cols: list[str],
    config: FeatureSelectionConfig,
) -> tuple[str, float]:
    model = _selector_model(task_type, config)
    train_pool = _make_pool(train, target_col, feature_cols, cat_cols)
    valid_pool = _make_pool(valid, target_col, feature_cols, cat_cols)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    pred = _predict_selector_model(model, valid, feature_cols, cat_cols, task_type)
    return _primary_metric(valid[target_col], pred, task_type)


def _candidate_ns(config: FeatureSelectionConfig, n_available: int) -> list[int]:
    if not config.auto_top_n:
        return [min(config.top_n, n_available)]
    raw = config.top_n_candidates or [config.top_n]
    values = sorted({min(int(n), n_available) for n in raw if int(n) > 0})
    return values or [min(config.top_n, n_available)]


def _choose_feature_set_by_valid(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target_col: str,
    task_type: str,
    candidate_sets: dict[int, list[str]],
    cat_cols: list[str],
    config: FeatureSelectionConfig,
) -> tuple[list[str], pd.DataFrame]:
    """Choose N only on selection-valid. Test/OOT is intentionally unavailable here."""
    rows = []
    for requested_n in sorted(candidate_sets):
        cols = _unique_keep_order(candidate_sets[requested_n])
        metric_name, value = _fit_and_score_feature_set(
            train=train,
            valid=valid,
            target_col=target_col,
            task_type=task_type,
            feature_cols=cols,
            cat_cols=[c for c in cat_cols if c in cols],
            config=config,
        )
        rows.append({
            "requested_n": int(requested_n),
            "n_features": int(len(cols)),
            "valid_metric": metric_name,
            "valid_value": float(value),
        })

    table = pd.DataFrame(rows).sort_values("n_features").reset_index(drop=True)
    metric_name = str(table.iloc[0]["valid_metric"])
    higher = _metric_higher_is_better(metric_name)
    best_value = table["valid_value"].max() if higher else table["valid_value"].min()
    tol = float(config.top_n_tolerance)
    if higher:
        eligible = table[table["valid_value"] >= best_value - tol]
    else:
        eligible = table[table["valid_value"] <= best_value + tol]
    # If quality is tied (or inside tolerance), prefer fewer features.
    chosen_row = eligible.sort_values(["n_features", "valid_value"], ascending=[True, not higher]).iloc[0]
    chosen_n = int(chosen_row["requested_n"])
    table["selected_n"] = table["requested_n"].eq(chosen_n)
    table["best_valid_value"] = float(best_value)
    table["top_n_tolerance"] = tol
    return _unique_keep_order(candidate_sets[chosen_n]), table


def _shap_calc_type(name: str):
    name = str(name).lower().strip()
    mapping = {
        "approximate": EShapCalcType.Approximate,
        "regular": EShapCalcType.Regular,
        "exact": EShapCalcType.Exact,
    }
    if name not in mapping:
        raise ValueError(f"Unknown rfe_shap_calc_type={name!r}; choose one of {sorted(mapping)}")
    return mapping[name]

def select_by_importance(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target_col: str,
    task_type: str,
    feature_cols: list[str],
    cat_cols: list[str],
    config: FeatureSelectionConfig,
) -> tuple[list[str], pd.DataFrame]:
    protected = [c for c in (config.protected_features or []) if c in feature_cols]
    model = _selector_model(task_type, config)
    train_pool = _make_pool(train, target_col, feature_cols, cat_cols)
    valid_pool = _make_pool(valid, target_col, feature_cols, cat_cols)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    importance = model.get_feature_importance(type="PredictionValuesChange")
    imp = pd.DataFrame({"feature": feature_cols, "importance": importance}).sort_values("importance", ascending=False)
    imp["importance_rank"] = np.arange(1, len(imp) + 1)
    ranked = imp["feature"].tolist()

    candidate_sets = {
        n: _unique_keep_order(protected + ranked[:n])
        for n in _candidate_ns(config, len(feature_cols))
    }
    if config.auto_top_n and len(candidate_sets) > 1:
        selected, n_search = _choose_feature_set_by_valid(
            train, valid, target_col, task_type, candidate_sets, cat_cols, config
        )
        imp.attrs["top_n_search"] = n_search.to_dict(orient="records")
        imp.attrs["selected_top_n"] = int(n_search.loc[n_search["selected_n"], "requested_n"].iloc[0])
    else:
        selected = next(iter(candidate_sets.values()))
        imp.attrs["selected_top_n"] = int(next(iter(candidate_sets.keys())))
    return selected, imp


def _rfe_algorithm(name: str):
    name = str(name).lower().strip()
    mapping = {
        "prediction_values_change": EFeaturesSelectionAlgorithm.RecursiveByPredictionValuesChange,
        "shap_values": EFeaturesSelectionAlgorithm.RecursiveByShapValues,
        "loss_function_change": EFeaturesSelectionAlgorithm.RecursiveByLossFunctionChange,
    }
    if name not in mapping:
        raise ValueError(f"Unknown rfe_algorithm={name!r}; choose one of {sorted(mapping)}")
    return mapping[name]


def select_by_rfe(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target_col: str,
    task_type: str,
    feature_cols: list[str],
    cat_cols: list[str],
    config: FeatureSelectionConfig,
) -> tuple[list[str], pd.DataFrame]:
    protected = [c for c in (config.protected_features or []) if c in feature_cols]
    candidate_ns = _candidate_ns(config, len(feature_cols))
    min_n = min(candidate_ns)

    if len(feature_cols) <= min_n:
        report = pd.DataFrame({"feature": feature_cols, "rfe_selected": True})
        report.attrs["selected_top_n"] = int(len(feature_cols))
        return list(feature_cols), report

    model = _selector_model(task_type, config)
    train_pool = _make_pool(train, target_col, feature_cols, cat_cols)
    valid_pool = _make_pool(valid, target_col, feature_cols, cat_cols)

    result = model.select_features(
        train_pool,
        eval_set=valid_pool,
        features_for_select=list(range(len(feature_cols))),
        num_features_to_select=min_n,
        algorithm=_rfe_algorithm(config.rfe_algorithm),
        steps=min(config.rfe_steps, max(len(feature_cols) - min_n, 1)),
        shap_calc_type=_shap_calc_type(config.rfe_shap_calc_type),
        train_final_model=False,
        logging_level="Silent",
        plot=False,
    )

    eliminated = result.get("eliminated_features_names", [])
    if not eliminated:
        eliminated_idx = result.get("eliminated_features", [])
        eliminated = [feature_cols[i] for i in eliminated_idx]

    # CatBoost returns elimination order. Therefore one RFE pass down to min(N-grid)
    # is enough to reconstruct every larger nested feature set.
    candidate_sets = {}
    for n in candidate_ns:
        remove_count = max(len(feature_cols) - int(n), 0)
        removed = set(eliminated[:remove_count])
        cols = [c for c in feature_cols if c not in removed]
        candidate_sets[int(n)] = _unique_keep_order(protected + cols)

    if config.auto_top_n and len(candidate_sets) > 1:
        selected, n_search = _choose_feature_set_by_valid(
            train, valid, target_col, task_type, candidate_sets, cat_cols, config
        )
        selected_top_n = int(n_search.loc[n_search["selected_n"], "requested_n"].iloc[0])
    else:
        selected_top_n = int(candidate_ns[-1] if not config.auto_top_n else candidate_ns[0])
        # Non-auto mode has one candidate. This also keeps behaviour deterministic.
        selected = candidate_sets[int(candidate_ns[0])]
        n_search = None

    order = {name: i + 1 for i, name in enumerate(eliminated)}
    report = pd.DataFrame({"feature": feature_cols})
    report["rfe_selected"] = report["feature"].isin(selected)
    report["rfe_elimination_order"] = report["feature"].map(order)
    report.attrs["selected_top_n"] = selected_top_n
    report.attrs["rfe_loss_graph"] = result.get("loss_graph")
    if n_search is not None:
        report.attrs["top_n_search"] = n_search.to_dict(orient="records")
    return selected, report


def _metric_rfe_sizes(n_features: int, config: FeatureSelectionConfig) -> list[int]:
    """Build an adaptive elimination path; final N is chosen by metric, not this path."""
    min_n = min(int(config.metric_rfe_min_features), int(n_features))
    sizes = [int(n_features)]
    while sizes[-1] > min_n:
        current = sizes[-1]
        drop = max(int(config.metric_rfe_min_drop), int(round(current * config.metric_rfe_drop_fraction)))
        nxt = max(min_n, current - drop)
        if nxt >= current:
            break
        sizes.append(int(nxt))

    # Computational cap on number of model evaluations, never a feature-count cutoff.
    if len(sizes) > int(config.metric_rfe_max_models):
        idx = np.linspace(0, len(sizes) - 1, int(config.metric_rfe_max_models)).round().astype(int)
        sizes = [sizes[i] for i in sorted(set(idx))]
        if sizes[-1] != min_n:
            sizes.append(min_n)
    return _unique_keep_order(sizes)


def select_by_metric_rfe(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target_col: str,
    task_type: str,
    feature_cols: list[str],
    cat_cols: list[str],
    config: FeatureSelectionConfig,
) -> tuple[list[str], pd.DataFrame]:
    """
    RFE ranking + validation-driven cutoff.

    No top-N is supplied. We walk a nested RFE path and choose the smallest feature set
    that satisfies the configured valid-quality rule. `one_se` uses bootstrap uncertainty
    of the best validation metric and is the recommended default.
    """
    protected = [c for c in (config.protected_features or []) if c in feature_cols]
    if not feature_cols:
        raise ValueError("metric_rfe received no features")

    sizes = _metric_rfe_sizes(len(feature_cols), config)
    min_n = min(sizes)

    if len(feature_cols) <= min_n:
        report = pd.DataFrame({"feature": feature_cols, "metric_rfe_selected": True})
        report.attrs["metric_rfe_curve"] = [{
            "requested_n": len(feature_cols), "n_features": len(feature_cols),
            "valid_metric": None, "valid_value": np.nan, "valid_se": np.nan,
            "selected_cutoff": True,
        }]
        report.attrs["metric_rfe_selected_n"] = len(feature_cols)
        report.attrs["metric_rfe_rule"] = config.metric_rfe_rule
        return list(feature_cols), report

    # One CatBoost RFE pass provides an elimination order down to the technical floor.
    rank_model = _selector_model(task_type, config)
    train_pool = _make_pool(train, target_col, feature_cols, cat_cols)
    valid_pool = _make_pool(valid, target_col, feature_cols, cat_cols)
    rfe_result = rank_model.select_features(
        train_pool,
        eval_set=valid_pool,
        features_for_select=list(range(len(feature_cols))),
        num_features_to_select=min_n,
        algorithm=_rfe_algorithm(config.rfe_algorithm),
        steps=min(config.rfe_steps, max(len(feature_cols) - min_n, 1)),
        shap_calc_type=_shap_calc_type(config.rfe_shap_calc_type),
        train_final_model=False,
        logging_level="Silent",
        plot=False,
    )
    eliminated = rfe_result.get("eliminated_features_names", [])
    if not eliminated:
        eliminated_idx = rfe_result.get("eliminated_features", [])
        eliminated = [feature_cols[i] for i in eliminated_idx]

    candidate_sets: dict[int, list[str]] = {}
    for n in sizes:
        remove_count = max(len(feature_cols) - int(n), 0)
        removed = set(eliminated[:remove_count])
        candidate_sets[int(n)] = _unique_keep_order(protected + [c for c in feature_cols if c not in removed])

    eval_df = valid
    if len(eval_df) > int(config.metric_rfe_eval_rows):
        eval_df = eval_df.sample(int(config.metric_rfe_eval_rows), random_state=config.selector_random_seed)

    curve_rows = []
    for i, n in enumerate(sizes):
        cols = candidate_sets[int(n)]
        model = _selector_model(task_type, config)
        tr_pool = _make_pool(train, target_col, cols, [c for c in cat_cols if c in cols])
        va_pool = _make_pool(valid, target_col, cols, [c for c in cat_cols if c in cols])
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
        pred = _predict_selector_model(model, eval_df, cols, [c for c in cat_cols if c in cols], task_type)
        metric_name, value = _primary_metric(eval_df[target_col], pred, task_type)
        se, _ = _bootstrap_metric_se(
            eval_df[target_col], pred, metric_name,
            repeats=config.metric_rfe_bootstrap_repeats,
            random_seed=config.selector_random_seed + i * 1009,
        )
        curve_rows.append({
            "requested_n": int(n),
            "n_features": int(len(cols)),
            "valid_metric": metric_name,
            "valid_value": float(value),
            "valid_se": float(se),
        })

    curve = pd.DataFrame(curve_rows).sort_values("n_features").reset_index(drop=True)
    metric_name = str(curve.iloc[0]["valid_metric"])
    higher = _metric_higher_is_better(metric_name)
    best_idx = curve["valid_value"].idxmax() if higher else curve["valid_value"].idxmin()
    best_row = curve.loc[best_idx]
    best_value = float(best_row["valid_value"])
    best_se = float(best_row["valid_se"]) if pd.notna(best_row["valid_se"]) else 0.0

    rule = config.metric_rfe_rule
    if rule == "best":
        eligible = curve[curve.index == best_idx]
        margin = 0.0
    elif rule == "tolerance":
        margin = float(config.metric_rfe_tolerance)
        if higher:
            eligible = curve[curve["valid_value"] >= best_value - margin]
        else:
            eligible = curve[curve["valid_value"] <= best_value + margin]
    else:  # one_se
        margin = float(config.metric_rfe_one_se_multiplier) * best_se
        if higher:
            eligible = curve[curve["valid_value"] >= best_value - margin]
        else:
            eligible = curve[curve["valid_value"] <= best_value + margin]

    # Prefer the smallest statistically/operationally equivalent model.
    chosen = eligible.sort_values("n_features", ascending=True).iloc[0]
    chosen_requested_n = int(chosen["requested_n"])
    selected = candidate_sets[chosen_requested_n]
    curve["selected_cutoff"] = curve["requested_n"].eq(chosen_requested_n)
    curve["best_valid_value"] = best_value
    curve["best_valid_se"] = best_se
    curve["selection_margin"] = margin
    curve["metric_rfe_rule"] = rule

    order = {name: i + 1 for i, name in enumerate(eliminated)}
    report = pd.DataFrame({"feature": feature_cols})
    report["metric_rfe_selected"] = report["feature"].isin(selected)
    report["rfe_elimination_order"] = report["feature"].map(order)
    report.attrs["metric_rfe_curve"] = curve.to_dict(orient="records")
    report.attrs["metric_rfe_selected_n"] = int(len(selected))
    report.attrs["metric_rfe_rule"] = rule
    report.attrs["metric_rfe_best_value"] = best_value
    report.attrs["metric_rfe_best_se"] = best_se
    report.attrs["metric_rfe_selection_margin"] = margin
    report.attrs["rfe_loss_graph"] = rfe_result.get("loss_graph")
    return selected, report


def select_by_permutation(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target_col: str,
    task_type: str,
    feature_cols: list[str],
    cat_cols: list[str],
    config: FeatureSelectionConfig,
) -> tuple[list[str], pd.DataFrame]:
    """Rank features by actual valid-metric degradation after shuffling a column."""
    protected = [c for c in (config.protected_features or []) if c in feature_cols]
    working = list(feature_cols)
    prescreened_out: list[str] = []

    max_perm = config.permutation_max_features
    if max_perm is not None and len(working) > int(max_perm):
        quick_cfg = replace(config, auto_top_n=False, top_n=int(max_perm))
        preselected, pre_imp = select_by_importance(
            train, valid, target_col, task_type, working, cat_cols, quick_cfg
        )
        prescreened_out = [c for c in working if c not in set(preselected)]
        working = preselected

    model = _selector_model(task_type, config)
    train_pool = _make_pool(train, target_col, working, [c for c in cat_cols if c in working])
    valid_pool = _make_pool(valid, target_col, working, [c for c in cat_cols if c in working])
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    eval_df = valid
    if len(eval_df) > config.permutation_eval_rows:
        eval_df = eval_df.sample(config.permutation_eval_rows, random_state=config.selector_random_seed)
    eval_cat = [c for c in cat_cols if c in working]
    baseline_pred = _predict_selector_model(model, eval_df, working, eval_cat, task_type)
    metric_name, baseline = _primary_metric(eval_df[target_col], baseline_pred, task_type)
    higher = _metric_higher_is_better(metric_name)

    rng = np.random.default_rng(config.selector_random_seed)
    rows = []
    base_X = eval_df[working].copy()
    for col in working:
        drops = []
        perm_values = base_X[col].to_numpy(copy=True)
        for _ in range(config.permutation_repeats):
            shuffled = base_X.copy()
            shuffled[col] = rng.permutation(perm_values)
            pred = _predict_selector_model(model, shuffled, working, eval_cat, task_type)
            _, perm_value = _primary_metric(eval_df[target_col], pred, task_type)
            # Positive importance always means "quality became worse after permutation".
            delta = (baseline - perm_value) if higher else (perm_value - baseline)
            drops.append(float(delta))
        rows.append({
            "feature": col,
            "permutation_importance": float(np.mean(drops)),
            "permutation_importance_std": float(np.std(drops, ddof=0)),
            "permutation_baseline_metric": metric_name,
            "permutation_baseline_value": float(baseline),
            "permutation_evaluated": True,
        })

    report = pd.DataFrame(rows).sort_values("permutation_importance", ascending=False)
    report["permutation_rank"] = np.arange(1, len(report) + 1)
    if prescreened_out:
        report = pd.concat([
            report,
            pd.DataFrame({
                "feature": prescreened_out,
                "permutation_importance": np.nan,
                "permutation_importance_std": np.nan,
                "permutation_baseline_metric": metric_name,
                "permutation_baseline_value": baseline,
                "permutation_evaluated": False,
                "permutation_rank": np.nan,
            })
        ], ignore_index=True)

    ranked = report.loc[report["permutation_evaluated"], "feature"].tolist()
    candidate_sets = {
        n: _unique_keep_order(protected + ranked[: min(n, len(ranked))])
        for n in _candidate_ns(config, len(ranked))
    }
    if config.auto_top_n and len(candidate_sets) > 1:
        selected, n_search = _choose_feature_set_by_valid(
            train, valid, target_col, task_type, candidate_sets, cat_cols, config
        )
        report.attrs["top_n_search"] = n_search.to_dict(orient="records")
        report.attrs["selected_top_n"] = int(n_search.loc[n_search["selected_n"], "requested_n"].iloc[0])
    else:
        selected = next(iter(candidate_sets.values()))
        report.attrs["selected_top_n"] = int(next(iter(candidate_sets.keys())))
    report.attrs["permutation_metric"] = metric_name
    report.attrs["permutation_baseline"] = float(baseline)
    report.attrs["permutation_eval_rows"] = int(len(eval_df))
    return selected, report


def select_by_permutation_ci(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target_col: str,
    task_type: str,
    feature_cols: list[str],
    cat_cols: list[str],
    config: FeatureSelectionConfig,
) -> tuple[list[str], pd.DataFrame]:
    """Permutation selection where CI, rather than top-N, decides whether a feature survives."""
    protected = [c for c in (config.protected_features or []) if c in feature_cols]
    working = list(feature_cols)
    prescreened_out: list[str] = []

    max_perm = config.permutation_ci_max_features
    if max_perm is not None and len(working) > int(max_perm):
        quick_cfg = replace(config, auto_top_n=False, top_n=int(max_perm))
        working, _ = select_by_importance(
            train, valid, target_col, task_type, working, cat_cols, quick_cfg
        )
        prescreened_out = [c for c in feature_cols if c not in set(working)]

    model = _selector_model(task_type, config)
    train_pool = _make_pool(train, target_col, working, [c for c in cat_cols if c in working])
    valid_pool = _make_pool(valid, target_col, working, [c for c in cat_cols if c in working])
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    eval_df = valid
    if len(eval_df) > config.permutation_eval_rows:
        eval_df = eval_df.sample(config.permutation_eval_rows, random_state=config.selector_random_seed)
    eval_cat = [c for c in cat_cols if c in working]
    y = np.asarray(eval_df[target_col])
    baseline_pred = _predict_selector_model(model, eval_df, working, eval_cat, task_type)
    metric_name, baseline = _primary_metric(y, baseline_pred, task_type)

    rng = np.random.default_rng(config.selector_random_seed)
    alpha = (1.0 - float(config.permutation_ci_confidence)) / 2.0
    base_X = eval_df[working].copy()
    rows = []

    for feature_no, col in enumerate(working):
        perm_values = base_X[col].to_numpy(copy=True)
        overall_deltas = []
        bootstrap_deltas = []
        for repeat_no in range(config.permutation_repeats):
            shuffled = base_X.copy()
            shuffled[col] = rng.permutation(perm_values)
            perm_pred = _predict_selector_model(model, shuffled, working, eval_cat, task_type)
            perm_metric = _metric_from_name(y, perm_pred, metric_name)
            overall_deltas.append(_quality_delta(baseline, perm_metric, metric_name))

            # Paired bootstrap: baseline and permuted predictions use the same sampled rows.
            n = len(y)
            b_rows = min(int(config.permutation_ci_bootstrap_rows), n)
            for _ in range(int(config.permutation_ci_bootstrap_repeats)):
                idx = rng.integers(0, n, size=b_rows)
                base_b = _metric_from_name(y[idx], baseline_pred[idx], metric_name)
                perm_b = _metric_from_name(y[idx], perm_pred[idx], metric_name)
                if np.isfinite(base_b) and np.isfinite(perm_b):
                    bootstrap_deltas.append(_quality_delta(base_b, perm_b, metric_name))

        mean_imp = float(np.mean(overall_deltas)) if overall_deltas else np.nan
        if bootstrap_deltas:
            ci_low = float(np.quantile(bootstrap_deltas, alpha))
            ci_high = float(np.quantile(bootstrap_deltas, 1.0 - alpha))
            positive_rate = float(np.mean(np.asarray(bootstrap_deltas) > config.permutation_ci_min_importance))
        else:
            ci_low = ci_high = np.nan
            positive_rate = np.nan
        passed = bool(pd.notna(ci_low) and ci_low > config.permutation_ci_min_importance)
        rows.append({
            "feature": col,
            "permutation_importance": mean_imp,
            "permutation_importance_std": float(np.std(overall_deltas, ddof=1)) if len(overall_deltas) > 1 else 0.0,
            "permutation_ci_low": ci_low,
            "permutation_ci_high": ci_high,
            "permutation_ci_positive_rate": positive_rate,
            "permutation_ci_pass": passed,
            "permutation_baseline_metric": metric_name,
            "permutation_baseline_value": float(baseline),
            "permutation_evaluated": True,
        })

    report = pd.DataFrame(rows).sort_values(
        ["permutation_ci_pass", "permutation_ci_low", "permutation_importance"],
        ascending=[False, False, False], na_position="last"
    )
    report["permutation_rank"] = np.arange(1, len(report) + 1)
    selected_signal = report.loc[report["permutation_ci_pass"], "feature"].tolist()
    fallback = False
    if not selected_signal:
        if config.permutation_ci_fallback_features is None:
            raise ValueError(
                "Permutation-CI did not find any statistically positive feature. "
                "Increase sample/repeats, relax confidence, protect critical features, "
                "or explicitly set permutation_ci_fallback_features if a fallback is desired."
            )
        fallback = True
        positive = report[report["permutation_importance"] > config.permutation_ci_min_importance]
        source = positive if not positive.empty else report
        k = min(int(config.permutation_ci_fallback_features), len(source))
        selected_signal = source.head(k)["feature"].tolist()

    if prescreened_out:
        report = pd.concat([report, pd.DataFrame({
            "feature": prescreened_out,
            "permutation_importance": np.nan,
            "permutation_importance_std": np.nan,
            "permutation_ci_low": np.nan,
            "permutation_ci_high": np.nan,
            "permutation_ci_positive_rate": np.nan,
            "permutation_ci_pass": False,
            "permutation_baseline_metric": metric_name,
            "permutation_baseline_value": baseline,
            "permutation_evaluated": False,
            "permutation_rank": np.nan,
        })], ignore_index=True)

    selected = _unique_keep_order(protected + selected_signal)
    report["permutation_ci_selected"] = report["feature"].isin(selected)
    report.attrs["permutation_metric"] = metric_name
    report.attrs["permutation_baseline"] = float(baseline)
    report.attrs["permutation_eval_rows"] = int(len(eval_df))
    report.attrs["permutation_ci_confidence"] = float(config.permutation_ci_confidence)
    report.attrs["permutation_ci_threshold"] = float(config.permutation_ci_min_importance)
    report.attrs["permutation_ci_fallback"] = fallback
    report.attrs["permutation_ci_selected_n"] = int(len(selected))
    return selected, report


def select_by_null_importance(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target_col: str,
    task_type: str,
    feature_cols: list[str],
    cat_cols: list[str],
    config: FeatureSelectionConfig,
) -> tuple[list[str], pd.DataFrame]:
    """Select features whose real CatBoost importance exceeds their shuffled-target null distribution."""
    protected = [c for c in (config.protected_features or []) if c in feature_cols]
    working = list(feature_cols)
    prescreened_out: list[str] = []

    max_null = config.null_importance_max_features
    if max_null is not None and len(working) > int(max_null):
        quick_cfg = replace(config, auto_top_n=False, top_n=int(max_null))
        working, _ = select_by_importance(
            train, valid, target_col, task_type, working, cat_cols, quick_cfg
        )
        prescreened_out = [c for c in feature_cols if c not in set(working)]

    # Use the same fixed-iteration training protocol for real and null targets.
    fixed_cfg = replace(config, selector_early_stopping_rounds=None)
    real_model = _selector_model(task_type, fixed_cfg)
    real_pool = _make_pool(train, target_col, working, [c for c in cat_cols if c in working])
    real_model.fit(real_pool, verbose=False)
    real_importance = np.asarray(real_model.get_feature_importance(type="PredictionValuesChange"), dtype=float)

    rng = np.random.default_rng(config.selector_random_seed)
    null_matrix = np.zeros((int(config.null_importance_repeats), len(working)), dtype=float)
    y = train[target_col].to_numpy(copy=True)
    for i in range(int(config.null_importance_repeats)):
        null_train = train.copy()
        null_train[target_col] = rng.permutation(y)
        null_model = _selector_model(task_type, fixed_cfg)
        null_pool = _make_pool(null_train, target_col, working, [c for c in cat_cols if c in working])
        null_model.fit(null_pool, verbose=False)
        null_matrix[i] = np.asarray(
            null_model.get_feature_importance(type="PredictionValuesChange"), dtype=float
        )

    q = float(config.null_importance_quantile)
    null_cutoff = np.quantile(null_matrix, q, axis=0)
    null_mean = np.mean(null_matrix, axis=0)
    null_std = np.std(null_matrix, axis=0, ddof=1) if null_matrix.shape[0] > 1 else np.zeros(len(working))
    score = (real_importance - null_mean) / np.where(null_std > 1e-12, null_std, 1e-12)
    passed = real_importance > null_cutoff

    report = pd.DataFrame({
        "feature": working,
        "real_importance": real_importance,
        "null_importance_mean": null_mean,
        "null_importance_std": null_std,
        "null_importance_cutoff": null_cutoff,
        "null_importance_score": score,
        "null_importance_pass": passed,
        "null_importance_evaluated": True,
    }).sort_values(
        ["null_importance_pass", "null_importance_score", "real_importance"],
        ascending=[False, False, False], na_position="last"
    )

    selected_signal = report.loc[report["null_importance_pass"], "feature"].tolist()
    fallback = False
    if not selected_signal:
        if config.null_importance_fallback_features is None:
            raise ValueError(
                "Null-importance did not find any feature above the noise cutoff. "
                "Increase null_importance_repeats, lower null_importance_quantile, "
                "protect critical features, or explicitly configure a fallback."
            )
        fallback = True
        k = min(int(config.null_importance_fallback_features), len(report))
        selected_signal = report.head(k)["feature"].tolist()

    if prescreened_out:
        report = pd.concat([report, pd.DataFrame({
            "feature": prescreened_out,
            "real_importance": np.nan,
            "null_importance_mean": np.nan,
            "null_importance_std": np.nan,
            "null_importance_cutoff": np.nan,
            "null_importance_score": np.nan,
            "null_importance_pass": False,
            "null_importance_evaluated": False,
        })], ignore_index=True)

    selected = _unique_keep_order(protected + selected_signal)
    report["null_importance_selected"] = report["feature"].isin(selected)
    report.attrs["null_importance_quantile"] = q
    report.attrs["null_importance_repeats"] = int(config.null_importance_repeats)
    report.attrs["null_importance_fallback"] = fallback
    report.attrs["null_importance_selected_n"] = int(len(selected))
    return selected, report


def _stability_base_selector(
    base_method: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target_col: str,
    task_type: str,
    feature_cols: list[str],
    cat_cols: list[str],
    config: FeatureSelectionConfig,
) -> tuple[list[str], pd.DataFrame]:
    base_method = str(base_method).lower().strip()
    fold_cfg = replace(
        config,
        auto_top_n=False,
        top_n=min(config.stability_top_n_per_fold, len(feature_cols)),
    )
    if base_method == "importance":
        return select_by_importance(train, valid, target_col, task_type, feature_cols, cat_cols, fold_cfg)
    if base_method == "permutation":
        return select_by_permutation(train, valid, target_col, task_type, feature_cols, cat_cols, fold_cfg)
    if base_method == "permutation_ci":
        return select_by_permutation_ci(train, valid, target_col, task_type, feature_cols, cat_cols, fold_cfg)
    if base_method == "metric_rfe":
        fold_cfg = replace(fold_cfg, rfe_algorithm="loss_function_change")
        return select_by_metric_rfe(train, valid, target_col, task_type, feature_cols, cat_cols, fold_cfg)
    if base_method in {"rfe_loss", "rfe_shap", "rfe_pvc"}:
        algo = {
            "rfe_loss": "loss_function_change",
            "rfe_shap": "shap_values",
            "rfe_pvc": "prediction_values_change",
        }[base_method]
        fold_cfg = replace(fold_cfg, rfe_algorithm=algo)
        return select_by_rfe(train, valid, target_col, task_type, feature_cols, cat_cols, fold_cfg)
    raise ValueError(
        "stability_base_method must be one of: importance, permutation, permutation_ci, metric_rfe, rfe_loss, rfe_shap, rfe_pvc"
    )


def select_by_stability(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target_col: str,
    task_type: str,
    feature_cols: list[str],
    cat_cols: list[str],
    month_col: str,
    config: FeatureSelectionConfig,
    apply_psi_per_fold: bool = False,
) -> tuple[list[str], pd.DataFrame]:
    """Rolling temporal stability selection: select features repeatedly on historical folds."""
    if month_col not in train.columns or month_col not in valid.columns:
        raise ValueError("stability selection requires month_col in train and valid")

    history = pd.concat([train, valid], ignore_index=True)
    month_values = pd.to_datetime(history[month_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    history = history.assign(__fs_month=month_values)
    months = sorted(history["__fs_month"].dropna().unique())
    possible_valid = months[config.stability_min_train_months:]
    fold_months = possible_valid[-config.stability_n_splits:]
    if not fold_months:
        raise ValueError(
            "Not enough months for stability selection. "
            "Reduce stability_min_train_months/stability_n_splits or provide more history."
        )

    selected_sets = []
    rank_rows = []
    fold_meta = []
    for fold_no, val_month in enumerate(fold_months, start=1):
        fold_train = history[history["__fs_month"] < val_month].drop(columns="__fs_month")
        fold_valid = history[history["__fs_month"] == val_month].drop(columns="__fs_month")
        if fold_train.empty or fold_valid.empty:
            continue

        fold_features, _ = prefilter_features(fold_train, feature_cols, config, cat_cols=cat_cols)
        if apply_psi_per_fold and fold_features:
            psi = compute_psi_table(fold_train, fold_valid, fold_features, month_col, config)
            psi_pass = set(psi.loc[psi["psi_pass"], "feature"])
            exempt = set(config.psi_exempt_features or [])
            protected = set(config.protected_features or [])
            fold_features = [c for c in fold_features if c in psi_pass or c in exempt or c in protected]
        if not fold_features:
            continue

        fold_selected, fold_report = _stability_base_selector(
            config.stability_base_method,
            fold_train,
            fold_valid,
            target_col,
            task_type,
            fold_features,
            [c for c in cat_cols if c in fold_features],
            config,
        )
        selected_sets.append(set(fold_selected))

        # Ranking is used only as a tie-breaker after selection frequency.
        if "importance_rank" in fold_report.columns:
            rank_map = fold_report.set_index("feature")["importance_rank"].to_dict()
        elif "permutation_rank" in fold_report.columns:
            rank_map = fold_report.set_index("feature")["permutation_rank"].to_dict()
        elif "rfe_elimination_order" in fold_report.columns:
            # Features surviving RFE are strongest; eliminated late are stronger than eliminated early.
            max_order = pd.to_numeric(fold_report["rfe_elimination_order"], errors="coerce").max()
            rank_map = {}
            for _, row in fold_report.iterrows():
                if bool(row.get("rfe_selected", False)):
                    rank_map[row["feature"]] = 1.0
                elif pd.notna(row.get("rfe_elimination_order")):
                    rank_map[row["feature"]] = float(max_order - row["rfe_elimination_order"] + 2)
        else:
            rank_map = {}
        for feature in feature_cols:
            rank_rows.append({
                "fold": int(fold_no),
                "valid_month": str(pd.Timestamp(val_month).date()),
                "feature": feature,
                "selected_in_fold": feature in selected_sets[-1],
                "fold_rank": rank_map.get(feature, np.nan),
            })
        fold_meta.append({
            "fold": int(fold_no),
            "valid_month": str(pd.Timestamp(val_month).date()),
            "train_rows": int(len(fold_train)),
            "valid_rows": int(len(fold_valid)),
            "n_candidates": int(len(fold_features)),
            "n_selected": int(len(fold_selected)),
        })

    if not selected_sets:
        raise ValueError("No valid temporal folds were produced for stability selection")

    rank_df = pd.DataFrame(rank_rows)
    report = (
        rank_df.groupby("feature", as_index=False)
        .agg(
            stability_selection_count=("selected_in_fold", "sum"),
            stability_selection_rate=("selected_in_fold", "mean"),
            stability_mean_rank=("fold_rank", "mean"),
        )
    )
    report["stability_n_folds"] = int(len(selected_sets))
    report["stability_pass"] = report["stability_selection_rate"] >= config.stability_selection_rate
    protected = [c for c in (config.protected_features or []) if c in feature_cols]
    report["protected"] = report["feature"].isin(protected)
    report = report.sort_values(
        ["stability_selection_rate", "stability_mean_rank"],
        ascending=[False, True],
        na_position="last",
    )

    stable = report.loc[report["stability_pass"], "feature"].tolist()
    if not stable:
        # No hidden Top-N fallback: keep all features tied at the best observed selection rate.
        # If even the best rate is zero, fail loudly instead of inventing an arbitrary N.
        best_rate = float(report["stability_selection_rate"].max()) if not report.empty else 0.0
        if best_rate <= 0:
            raise ValueError(
                "Stability selection produced zero selection rate for every feature. "
                "Relax stability_selection_rate or change the fold-level selector."
            )
        stable = report.loc[report["stability_selection_rate"] == best_rate, "feature"].tolist()
        report.attrs["stability_fallback"] = True
        report.attrs["stability_fallback_rate"] = best_rate
    else:
        report.attrs["stability_fallback"] = False

    selected = _unique_keep_order(protected + stable)
    if config.stability_max_features is not None and len(selected) > int(config.stability_max_features):
        ordered = _unique_keep_order(protected + report["feature"].tolist())
        selected = ordered[: max(int(config.stability_max_features), len(protected))]

    report["stability_selected"] = report["feature"].isin(selected)
    report.attrs["stability_folds"] = fold_meta
    report.attrs["stability_base_method"] = config.stability_base_method
    report.attrs["stability_threshold"] = float(config.stability_selection_rate)
    return selected, report


def select_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target_col: str,
    task_type: str,
    candidate_features: list[str],
    cat_cols: list[str],
    month_col: str,
    config: Optional[dict | FeatureSelectionConfig] = None,
) -> tuple[list[str], pd.DataFrame, dict]:
    """Run one feature-selection strategy and return selected list + auditable report."""
    cfg = normalize_feature_selection_config(config)
    protected = [c for c in (cfg.protected_features or []) if c in candidate_features]

    candidates, prefilter = prefilter_features(train, candidate_features, cfg, cat_cols=cat_cols)
    if not candidates:
        raise ValueError("No features left after prefilter")

    report = prefilter.copy()
    psi_table = pd.DataFrame(columns=["feature", "psi", "psi_pass"])
    method_table = pd.DataFrame(columns=["feature"])
    method_metadata: dict = {}

    working = list(candidates)
    n_after_psi = None

    if cfg.method in {
        "psi", "psi_importance", "psi_rfe", "psi_metric_rfe",
        "psi_permutation", "psi_permutation_ci", "psi_null_importance", "psi_stability"
    }:
        psi_table = compute_psi_table(train, valid, working, month_col, cfg)
        psi_exempt = set(cfg.psi_exempt_features or [])
        psi_table["psi_exempt"] = psi_table["feature"].isin(psi_exempt)
        psi_pass = set(psi_table.loc[psi_table["psi_pass"], "feature"])
        working = [c for c in working if c in psi_pass or c in protected or c in psi_exempt]
        if not working:
            raise ValueError(
                "No features left after PSI. Increase psi_threshold or protect critical features."
            )
        n_after_psi = int(len(working))

    if cfg.method in {"importance", "psi_importance"}:
        working, method_table = select_by_importance(
            train, valid, target_col, task_type, working,
            [c for c in cat_cols if c in working], cfg,
        )

    elif cfg.method in {"rfe", "psi_rfe"}:
        working, method_table = select_by_rfe(
            train, valid, target_col, task_type, working,
            [c for c in cat_cols if c in working], cfg,
        )

    elif cfg.method in {"metric_rfe", "psi_metric_rfe"}:
        working, method_table = select_by_metric_rfe(
            train, valid, target_col, task_type, working,
            [c for c in cat_cols if c in working], cfg,
        )

    elif cfg.method in {"permutation", "psi_permutation"}:
        working, method_table = select_by_permutation(
            train, valid, target_col, task_type, working,
            [c for c in cat_cols if c in working], cfg,
        )

    elif cfg.method in {"permutation_ci", "psi_permutation_ci"}:
        working, method_table = select_by_permutation_ci(
            train, valid, target_col, task_type, working,
            [c for c in cat_cols if c in working], cfg,
        )

    elif cfg.method in {"null_importance", "psi_null_importance"}:
        working, method_table = select_by_null_importance(
            train, valid, target_col, task_type, working,
            [c for c in cat_cols if c in working], cfg,
        )

    elif cfg.method in {"stability", "psi_stability"}:
        working, method_table = select_by_stability(
            train=train,
            valid=valid,
            target_col=target_col,
            task_type=task_type,
            feature_cols=working,
            cat_cols=[c for c in cat_cols if c in working],
            month_col=month_col,
            config=cfg,
            apply_psi_per_fold=(cfg.method == "psi_stability" and cfg.stability_apply_psi_per_fold),
        )

    elif cfg.method == "none":
        working = list(candidates)

    selected = _unique_keep_order(protected + working)

    for extra in [psi_table, method_table]:
        if not extra.empty:
            # Avoid duplicate protected columns from stability report.
            duplicate_cols = [c for c in extra.columns if c != "feature" and c in report.columns]
            if duplicate_cols:
                extra = extra.drop(columns=duplicate_cols)
            report = report.merge(extra, on="feature", how="left")
    report["selected"] = report["feature"].isin(selected)
    report["selection_method"] = cfg.method

    # DataFrame attrs contain method-specific audit details without changing public return types.
    for key in [
        "top_n_search", "selected_top_n", "rfe_loss_graph",
        "metric_rfe_curve", "metric_rfe_selected_n", "metric_rfe_rule",
        "metric_rfe_best_value", "metric_rfe_best_se", "metric_rfe_selection_margin",
        "permutation_metric", "permutation_baseline", "permutation_eval_rows",
        "permutation_ci_confidence", "permutation_ci_threshold",
        "permutation_ci_fallback", "permutation_ci_selected_n",
        "null_importance_quantile", "null_importance_repeats",
        "null_importance_fallback", "null_importance_selected_n",
        "stability_folds", "stability_base_method", "stability_threshold", "stability_fallback",
    ]:
        if key in method_table.attrs:
            method_metadata[key] = method_table.attrs[key]

    n_basic_prefilter = int((prefilter["prefilter_reason"].isna() | prefilter["prefilter_reason"].eq("high_correlation")).sum())
    n_dropped_correlated = int(prefilter["prefilter_reason"].eq("high_correlation").sum())
    metadata = {
        "config": asdict(cfg),
        "n_candidates": int(len(candidate_features)),
        "n_after_basic_prefilter": n_basic_prefilter,
        "n_dropped_correlated": n_dropped_correlated,
        "n_after_prefilter": int(len(candidates)),
        "correlation_threshold": float(cfg.correlation_threshold) if cfg.drop_correlated_features else None,
        "correlation_method": cfg.correlation_method if cfg.drop_correlated_features else None,
        "n_after_psi": n_after_psi,
        "n_selected": int(len(selected)),
        "selected_features": selected,
        **method_metadata,
    }
    return selected, report, metadata


def save_selection_report(
    report: pd.DataFrame,
    metadata: dict,
    model_key: str,
    config: Optional[dict | FeatureSelectionConfig] = None,
) -> dict[str, str]:
    cfg = normalize_feature_selection_config(config)
    out_dir = Path(cfg.report_dir) / model_key / cfg.method
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "feature_selection_report.csv"
    json_path = out_dir / "feature_selection_metadata.json"
    selected_path = out_dir / "selected_features.json"

    report.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(selected_path, "w", encoding="utf-8") as f:
        json.dump(metadata.get("selected_features", []), f, ensure_ascii=False, indent=2)

    return {
        "feature_selection_report": str(csv_path),
        "feature_selection_metadata": str(json_path),
        "selected_features": str(selected_path),
    }
