"""Benchmark current static feature set against dynamic feature-selection strategies."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional
import json

import numpy as np
import pandas as pd

from .training_workflow import TrainJobConfig, train_one_job
from .train_pipeline_workflow import train_cashback_connected_clean, train_loyalty_connection_clean


DEFAULT_N_GRID = [100, 150, 200, 250, 300]

# Practical methods: cheap baselines + dynamic no-fixed-N candidates.
DEFAULT_STRATEGIES = {
    "psi_020": {
        "method": "psi",
        "psi_threshold": 0.20,
        "psi_reference_months": 3,
    },
    "psi_importance_auto_n": {
        "method": "psi_importance",
        "psi_threshold": 0.20,
        "psi_reference_months": 3,
        "auto_top_n": True,
        "top_n_candidates": DEFAULT_N_GRID,
    },
    "psi_metric_rfe_1se": {
        "method": "psi_metric_rfe",
        "psi_threshold": 0.20,
        "psi_reference_months": 3,
        "rfe_algorithm": "loss_function_change",
        "rfe_steps": 5,
        "metric_rfe_rule": "one_se",
        "metric_rfe_min_features": 30,
        "metric_rfe_drop_fraction": 0.15,
        "metric_rfe_min_drop": 10,
        "metric_rfe_bootstrap_repeats": 100,
    },
    "psi_permutation_ci": {
        "method": "psi_permutation_ci",
        "psi_threshold": 0.20,
        "psi_reference_months": 3,
        "permutation_repeats": 3,
        "permutation_eval_rows": 50_000,
        "permutation_ci_bootstrap_repeats": 40,
        "permutation_ci_confidence": 0.95,
    },
    "psi_stability_importance": {
        "method": "psi_stability",
        "psi_threshold": 0.20,
        "psi_reference_months": 3,
        "stability_base_method": "importance",
        "stability_n_splits": 4,
        "stability_min_train_months": 3,
        "stability_selection_rate": 0.75,
        "stability_top_n_per_fold": 200,
        "stability_max_features": None,
    },
}


# CatBoost recursive methods. LossFunctionChange is the main RFE candidate;
# SHAP-RFE is intentionally separated because it is materially heavier.
RFE_STRATEGIES = {
    "psi_rfe_loss_auto_n": {
        "method": "psi_rfe",
        "psi_threshold": 0.20,
        "psi_reference_months": 3,
        "rfe_algorithm": "loss_function_change",
        "rfe_steps": 5,
        "auto_top_n": True,
        "top_n_candidates": DEFAULT_N_GRID,
    },
    "psi_rfe_shap_auto_n": {
        "method": "psi_rfe",
        "psi_threshold": 0.20,
        "psi_reference_months": 3,
        "rfe_algorithm": "shap_values",
        "rfe_shap_calc_type": "regular",
        "rfe_steps": 5,
        "auto_top_n": True,
        "top_n_candidates": DEFAULT_N_GRID,
    },
}

# Null-importance is intentionally separate because it trains several shuffled-target models.
NULL_IMPORTANCE_STRATEGIES = {
    "psi_null_importance": {
        "method": "psi_null_importance",
        "psi_threshold": 0.20,
        "psi_reference_months": 3,
        "null_importance_repeats": 20,
        "null_importance_quantile": 0.95,
        "null_importance_max_features": None,
    },
}

# Backward compatibility with the previous notebook flag.
RFE_STRATEGY = {
    "rfe_top200": {
        "method": "rfe",
        "top_n": 200,
        "rfe_steps": 5,
        "rfe_algorithm": "prediction_values_change",
    }
}


def _strategy_config(
    base: TrainJobConfig,
    strategy_name: str,
    input_path: str,
    feature_selection: Optional[dict],
    output_root: str | Path,
) -> TrainJobConfig:
    output_root = Path(output_root)
    stem = Path(base.model_filename).stem
    suffix = Path(base.model_filename).suffix or ".cbm"

    fs = None if feature_selection is None else dict(feature_selection)
    if fs is not None:
        fs.setdefault("report_dir", str(output_root / "selection_reports"))

    return replace(
        base,
        model_key=f"{base.model_key}__{strategy_name}",
        input_path=str(input_path),
        output_split_dir=str(output_root / base.model_key / strategy_name / "splits"),
        model_output_dir=str(output_root / base.model_key / strategy_name / "models"),
        model_filename=f"{stem}_{strategy_name}{suffix}",
        split_suffix=strategy_name,
        legacy_test_suffix=None,
        feature_selection=fs,
    )


def _summary_rows(strategy_name: str, result: dict, base_model_key: str) -> list[dict]:
    rows = []
    n_features = len(result.get("feature_cols", []))
    fs_meta = result.get("feature_selection_metadata") or {}
    fs_cfg = fs_meta.get("config") or {}
    for split, metrics in result.get("metrics", {}).items():
        row = {
            "base_model_key": base_model_key,
            "strategy": strategy_name,
            "split": split,
            "n_features": n_features,
            "elapsed_minutes": result.get("elapsed_seconds", np.nan) / 60.0,
            "train_rows": len(result.get("train", [])),
            "valid_rows": len(result.get("valid", [])),
            "test_rows": len(result.get("test", [])),
            "selection_method": fs_cfg.get("method", "current" if strategy_name == "current" else None),
            "psi_threshold": fs_cfg.get("psi_threshold", np.nan),
            "selected_top_n": fs_meta.get("selected_top_n", np.nan),
            "n_after_basic_prefilter": fs_meta.get("n_after_basic_prefilter", np.nan),
            "n_dropped_correlated": fs_meta.get("n_dropped_correlated", np.nan),
            "n_after_prefilter": fs_meta.get("n_after_prefilter", np.nan),
            "correlation_threshold": fs_meta.get("correlation_threshold", np.nan),
            "correlation_method": fs_meta.get("correlation_method"),
            "n_after_psi": fs_meta.get("n_after_psi", np.nan),
            "stability_base_method": fs_meta.get("stability_base_method"),
            "stability_threshold": fs_meta.get("stability_threshold", np.nan),
            "stability_n_folds": len(fs_meta.get("stability_folds") or []),
            "permutation_metric": fs_meta.get("permutation_metric"),
            "dynamic_cutoff_rule": fs_meta.get("metric_rfe_rule"),
            "dynamic_selected_n": (
                fs_meta.get("metric_rfe_selected_n")
                or fs_meta.get("permutation_ci_selected_n")
                or fs_meta.get("null_importance_selected_n")
            ),
            "metric_rfe_best_value": fs_meta.get("metric_rfe_best_value", np.nan),
            "metric_rfe_best_se": fs_meta.get("metric_rfe_best_se", np.nan),
            "metric_rfe_selection_margin": fs_meta.get("metric_rfe_selection_margin", np.nan),
            "permutation_ci_confidence": fs_meta.get("permutation_ci_confidence", np.nan),
            "permutation_ci_threshold": fs_meta.get("permutation_ci_threshold", np.nan),
            "null_importance_quantile": fs_meta.get("null_importance_quantile", np.nan),
            "null_importance_repeats": fs_meta.get("null_importance_repeats", np.nan),
        }
        top_n_search = fs_meta.get("top_n_search") or []
        if top_n_search:
            chosen = [x for x in top_n_search if x.get("selected_n")]
            if chosen:
                row["top_n_valid_value"] = chosen[0].get("valid_value", np.nan)
                row["top_n_valid_metric"] = chosen[0].get("valid_metric")
        row.update(metrics or {})
        rows.append(row)
    return rows


def add_deltas_vs_current(summary: pd.DataFrame, task_type: str) -> pd.DataFrame:
    """Add feature-count and primary-quality improvement vs current, split by split."""
    if summary.empty:
        return summary
    result = summary.copy()
    primary_metric = "roc_auc" if task_type == "classification" else "rmse"
    result["primary_metric"] = primary_metric
    result["primary_value"] = result.get(primary_metric, np.nan)
    result["current_primary_value"] = np.nan
    result["primary_improvement_vs_current"] = np.nan
    result["feature_reduction_vs_current"] = np.nan
    result["feature_reduction_share_vs_current"] = np.nan

    for split in result["split"].dropna().unique():
        mask = result["split"] == split
        current = result.loc[mask & (result["strategy"] == "current")]
        if current.empty:
            continue
        base_value = current.iloc[0].get(primary_metric, np.nan)
        base_n = current.iloc[0].get("n_features", np.nan)
        result.loc[mask, "current_primary_value"] = base_value
        result.loc[mask, "feature_reduction_vs_current"] = base_n - result.loc[mask, "n_features"]
        if pd.notna(base_n) and base_n > 0:
            result.loc[mask, "feature_reduction_share_vs_current"] = (
                base_n - result.loc[mask, "n_features"]
            ) / base_n
        if pd.notna(base_value):
            if task_type == "classification":
                result.loc[mask, "primary_improvement_vs_current"] = result.loc[mask, primary_metric] - base_value
            else:
                result.loc[mask, "primary_improvement_vs_current"] = base_value - result.loc[mask, primary_metric]

    result["better_than_current"] = result["primary_improvement_vs_current"] > 0
    result.loc[result["strategy"] == "current", "better_than_current"] = False
    return result


def benchmark_feature_selection(
    base_config: TrainJobConfig,
    baseline_input_path: str,
    candidate_input_path: str,
    strategies: Optional[dict[str, dict]] = None,
    include_rfe: bool = False,
    include_null_importance: bool = False,
    output_root: str = "reports/feature_selection_benchmark",
) -> tuple[dict[str, dict], pd.DataFrame]:
    """
    Compare current static fs_sel against dynamic selection from full candidate dataset.

    baseline_input_path: current *_fs_sel.csv (or current reduced parquet)
    candidate_input_path: full dataset before static feature selection

    Important: every selector sees only train + valid. Test/OOT is used only after the final
    model is trained, so the benchmark against current remains honest.
    """
    strategies = dict(strategies or DEFAULT_STRATEGIES)
    if include_rfe:
        strategies.update(RFE_STRATEGY)
    if include_null_importance:
        strategies.update(NULL_IMPORTANCE_STRATEGIES)

    results: dict[str, dict] = {}
    rows = []

    current_cfg = _strategy_config(
        base=base_config,
        strategy_name="current",
        input_path=baseline_input_path,
        feature_selection=None,
        output_root=output_root,
    )
    current_result = train_one_job(current_cfg)
    results["current"] = current_result
    rows.extend(_summary_rows("current", current_result, base_config.model_key))

    for strategy_name, fs_cfg in strategies.items():
        cfg = _strategy_config(
            base=base_config,
            strategy_name=strategy_name,
            input_path=candidate_input_path,
            feature_selection=fs_cfg,
            output_root=output_root,
        )
        result = train_one_job(cfg)
        results[strategy_name] = result
        rows.extend(_summary_rows(strategy_name, result, base_config.model_key))

    summary = add_deltas_vs_current(pd.DataFrame(rows), base_config.task_type)

    output_dir = Path(output_root) / base_config.model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "feature_selection_benchmark.csv"
    summary.to_csv(summary_path, index=False)

    selected = {name: res.get("feature_cols", []) for name, res in results.items()}
    with open(output_dir / "selected_features_by_strategy.json", "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    # Save top-N search curves and stability folds in one audit artifact.
    selection_details = {}
    for name, res in results.items():
        meta = res.get("feature_selection_metadata") or {}
        selection_details[name] = {
            "selected_top_n": meta.get("selected_top_n"),
            "top_n_search": meta.get("top_n_search"),
            "stability_folds": meta.get("stability_folds"),
            "stability_base_method": meta.get("stability_base_method"),
            "permutation_metric": meta.get("permutation_metric"),
            "metric_rfe_curve": meta.get("metric_rfe_curve"),
            "metric_rfe_rule": meta.get("metric_rfe_rule"),
            "metric_rfe_selected_n": meta.get("metric_rfe_selected_n"),
            "permutation_ci_confidence": meta.get("permutation_ci_confidence"),
            "permutation_ci_threshold": meta.get("permutation_ci_threshold"),
            "permutation_ci_selected_n": meta.get("permutation_ci_selected_n"),
            "null_importance_quantile": meta.get("null_importance_quantile"),
            "null_importance_repeats": meta.get("null_importance_repeats"),
            "null_importance_selected_n": meta.get("null_importance_selected_n"),
            "n_after_basic_prefilter": meta.get("n_after_basic_prefilter"),
            "n_dropped_correlated": meta.get("n_dropped_correlated"),
            "n_after_prefilter": meta.get("n_after_prefilter"),
            "correlation_threshold": meta.get("correlation_threshold"),
            "correlation_method": meta.get("correlation_method"),
        }
    with open(output_dir / "selection_details.json", "w", encoding="utf-8") as f:
        json.dump(selection_details, f, ensure_ascii=False, indent=2)

    # Sanity check: current and full candidate should represent the same rows for a fair benchmark.
    current_counts = (
        len(current_result.get("train", [])),
        len(current_result.get("valid", [])),
        len(current_result.get("test", [])),
    )
    for name, res in results.items():
        if name == "current":
            continue
        counts = (len(res.get("train", [])), len(res.get("valid", [])), len(res.get("test", [])))
        if counts != current_counts:
            print(
                f"WARNING benchmark row-count mismatch for {name}: "
                f"current={current_counts}, {name}={counts}. "
                "Проверь, что full dataset и *_fs_sel построены из одной версии данных."
            )

    print(f"benchmark saved: {summary_path}")
    return results, summary



def benchmark_train_pipeline_feature_selection(
    model_type: str,
    baseline_input_path: str,
    candidate_input_path: str,
    train_start: str,
    train_end_exclusive: str,
    valid_month: str,
    test_month: Optional[str] = None,
    strategies: Optional[dict[str, dict]] = None,
    include_null_importance: bool = False,
    output_root: str = "reports/feature_selection_benchmark/train_pipeline",
    model_overrides: Optional[dict] = None,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """Benchmark dynamic FS for the two special models from 52.1.train_pipeline.

    ``model_type``: ``cashback`` or ``loyalty_connection``.

    The current/static model and every dynamic strategy use the same training logic as
    52.1. For loyalty, the current baseline's ``valid_model`` client IDs are reused by
    every dynamic strategy, so ``valid_calib`` remains the same client holdout and is
    never used by feature selection.
    """
    model_type = str(model_type).lower().strip()
    aliases = {
        "cashback_connected": "cashback",
        "loyalty": "loyalty_connection",
        "loyalty_connect": "loyalty_connection",
    }
    model_type = aliases.get(model_type, model_type)
    if model_type not in {"cashback", "loyalty_connection"}:
        raise ValueError("model_type must be 'cashback' or 'loyalty_connection'")

    strategies = dict(strategies or DEFAULT_STRATEGIES)
    model_overrides = dict(model_overrides or {})
    if include_null_importance:
        strategies.update(NULL_IMPORTANCE_STRATEGIES)

    base_model_key = "cashback_connected" if model_type == "cashback" else "loyalty_connect"
    task_type = "regression" if model_type == "cashback" else "classification"
    output_dir = Path(output_root) / base_model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    rows: list[dict] = []

    def run_one(strategy_name: str, input_path: str, fs_cfg: Optional[dict], fixed_clients=None):
        fs = None if fs_cfg is None else dict(fs_cfg)
        if fs is not None:
            fs.setdefault("report_dir", str(Path(output_root) / "selection_reports"))

        model_dir = output_dir / strategy_name / "models"
        split_dir = output_dir / strategy_name / "splits"
        model_key = f"{base_model_key}__{strategy_name}"

        if model_type == "cashback":
            return train_cashback_connected_clean(
                data_path=input_path,
                train_start=train_start,
                train_end_exclusive=train_end_exclusive,
                valid_month=valid_month,
                test_month=test_month,
                model_dir=str(model_dir),
                feature_selection=fs,
                model_key=model_key,
                model_filename=f"cashback_connected_{strategy_name}.cbm",
                output_split_dir=str(split_dir),
                **model_overrides,
            )

        return train_loyalty_connection_clean(
            data_path=input_path,
            train_start=train_start,
            train_end_exclusive=train_end_exclusive,
            valid_month=valid_month,
            test_month=test_month,
            model_dir=str(model_dir),
            feature_selection=fs,
            model_key=model_key,
            model_filename=f"loyalty_connect_{strategy_name}.cbm",
            output_split_dir=str(split_dir),
            fixed_valid_model_clients=fixed_clients,
            calibrator_filename=f"isotonic_calibrator_{strategy_name}.pkl",
            **model_overrides,
        )

    # Current static feature set first.
    current_result = run_one("current", baseline_input_path, None)
    results["current"] = current_result
    rows.extend(_summary_rows("current", current_result, base_model_key))

    # Freeze loyalty's client-level valid split from the current implementation.
    fixed_valid_model_clients = None
    if model_type == "loyalty_connection":
        fixed_valid_model_clients = current_result["valid_model"]["cdi_id"].dropna().unique().tolist()

    for strategy_name, fs_cfg in strategies.items():
        result = run_one(
            strategy_name,
            candidate_input_path,
            fs_cfg,
            fixed_clients=fixed_valid_model_clients,
        )
        results[strategy_name] = result
        rows.extend(_summary_rows(strategy_name, result, base_model_key))

    summary = add_deltas_vs_current(pd.DataFrame(rows), task_type)
    summary_path = output_dir / "feature_selection_benchmark.csv"
    summary.to_csv(summary_path, index=False)

    selected = {name: res.get("feature_cols", []) for name, res in results.items()}
    with open(output_dir / "selected_features_by_strategy.json", "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    selection_details = {}
    for name, res in results.items():
        meta = res.get("feature_selection_metadata") or {}
        selection_details[name] = {
            "selected_top_n": meta.get("selected_top_n"),
            "top_n_search": meta.get("top_n_search"),
            "stability_folds": meta.get("stability_folds"),
            "stability_base_method": meta.get("stability_base_method"),
            "permutation_metric": meta.get("permutation_metric"),
            "metric_rfe_curve": meta.get("metric_rfe_curve"),
            "metric_rfe_rule": meta.get("metric_rfe_rule"),
            "metric_rfe_selected_n": meta.get("metric_rfe_selected_n"),
            "permutation_ci_confidence": meta.get("permutation_ci_confidence"),
            "permutation_ci_threshold": meta.get("permutation_ci_threshold"),
            "permutation_ci_selected_n": meta.get("permutation_ci_selected_n"),
            "null_importance_quantile": meta.get("null_importance_quantile"),
            "null_importance_repeats": meta.get("null_importance_repeats"),
            "null_importance_selected_n": meta.get("null_importance_selected_n"),
            "n_after_basic_prefilter": meta.get("n_after_basic_prefilter"),
            "n_dropped_correlated": meta.get("n_dropped_correlated"),
            "n_after_prefilter": meta.get("n_after_prefilter"),
            "correlation_threshold": meta.get("correlation_threshold"),
            "correlation_method": meta.get("correlation_method"),
            "selection_valid_role": meta.get("selection_valid_role"),
        }
    with open(output_dir / "selection_details.json", "w", encoding="utf-8") as f:
        json.dump(selection_details, f, ensure_ascii=False, indent=2)

    # Fairness checks: full-feature candidate must contain exactly the same rows as current.
    current_counts = (
        len(current_result.get("train", [])),
        len(current_result.get("valid", [])),
        len(current_result.get("test", [])),
    )
    for name, res in results.items():
        if name == "current":
            continue
        counts = (len(res.get("train", [])), len(res.get("valid", [])), len(res.get("test", [])))
        if counts != current_counts:
            print(
                f"WARNING benchmark row-count mismatch for {name}: "
                f"current={current_counts}, {name}={counts}. "
                "Rebuild *_full candidate from the current table keys."
            )
        if model_type == "loyalty_connection":
            cur_ids = set(current_result["valid_model"]["cdi_id"].dropna())
            dyn_ids = set(res["valid_model"]["cdi_id"].dropna())
            if cur_ids != dyn_ids:
                print(f"WARNING valid_model client mismatch for {name}: {len(cur_ids)} vs {len(dyn_ids)}")

    print(f"train_pipeline benchmark saved: {summary_path}")
    return results, summary

def best_strategy(summary: pd.DataFrame, prefer_split: str = "test") -> pd.DataFrame:
    """Compact ranking: quality first, then fewer features."""
    if summary.empty:
        return summary
    available = set(summary["split"].dropna())
    if prefer_split in available:
        split = prefer_split
    elif "valid" in available:
        split = "valid"
    elif "valid_calib" in available:
        split = "valid_calib"
    elif "valid_model_raw" in available:
        split = "valid_model_raw"
    else:
        split = next(iter(available))
    part = summary[summary["split"] == split].copy()
    return part.sort_values(
        ["primary_improvement_vs_current", "n_features"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)


def top_n_search_table(results: dict[str, dict]) -> pd.DataFrame:
    """Collect internal valid-only N-search curves from all strategies."""
    rows = []
    for strategy, res in results.items():
        meta = res.get("feature_selection_metadata") or {}
        for item in meta.get("top_n_search") or []:
            rows.append({"strategy": strategy, **item})
    return pd.DataFrame(rows)


def dynamic_cutoff_table(results: dict[str, dict]) -> pd.DataFrame:
    """Collect validation-only metric-RFE cutoff curves from benchmark results."""
    rows = []
    for strategy, res in results.items():
        meta = res.get("feature_selection_metadata") or {}
        for item in meta.get("metric_rfe_curve") or []:
            rows.append({"strategy": strategy, **item})
    return pd.DataFrame(rows)
