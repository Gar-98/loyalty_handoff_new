"""Общие функции для inference-ноутбука.

Сюда вынесены повторяющиеся SQL-генераторы, подготовка датафрейма,
загрузка моделей и скоринг, чтобы в ноутбуке оставалась только бизнес-логика.
"""

from pathlib import Path
import json
import os
import sys

import numpy as np
import pandas as pd


def find_project_root(marker="src/nbc_project", candidates=None):
    """Ищет корень проекта по папке src/nbc_project.

    Не использует Path.cwd() как единственный источник, потому что в Jupyter
    рабочая директория иногда удаляется/размонтируется и Path.cwd() падает.
    """
    candidates = list(candidates or [])

    try:
        candidates.append(Path.cwd())
    except FileNotFoundError:
        pass

    for base in [Path.home(), Path("/home/jovyan"), Path("/mnt/data"), Path("/")]:
        if base.exists():
            candidates.append(base)

    seen = set()
    for base in candidates:
        base = Path(base).expanduser()
        if base in seen or not base.exists():
            continue
        seen.add(base)

        if (base / marker).exists():
            return base

        # Ограничиваем поиск, чтобы не сканировать всю ФС слишком долго.
        for path in base.glob("**/src/nbc_project"):
            return path.parents[1]

    raise FileNotFoundError("Не найден корень проекта с папкой src/nbc_project")


def setup_project_path(project_root=None):
    """Добавляет src в sys.path и возвращает PROJECT_ROOT."""
    root = Path(project_root) if project_root is not None else find_project_root()
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    return root


def load_inference_config(project_root=None, config_path=None):
    """Читает configs/inference_config.json."""
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    path = Path(config_path) if config_path is not None else root / "configs" / "inference_config.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_inference_table_names(postfix):
    """Возвращает названия промежуточных таблиц inference-витрины."""
    return {
        "cb_base": f"sandbox_crft.cb_base_client_month_{postfix}",
        "cb_charges_compact": "sandbox_crft.cb_charges_compact_04_06_cashback_total_pred",
        "cb_category_spend_wide": f"sandbox_crft.cb_category_spend_wide_{postfix}",
        "cb_offered_percent_wide": f"sandbox_crft.cb_offered_percent_wide_{postfix}",
        "cb_selected_history_wide": f"sandbox_crft.cb_selected_history_wide_{postfix}",
        "cb_acrm_monthly": f"sandbox_crft.cb_acrm_monthly_{postfix}",
        "cb_acrm_targets": f"sandbox_crft.cb_acrm_targets_{postfix}",
        "cb_acrm_lags": f"sandbox_crft.cb_acrm_lags_{postfix}",
        "cb_activity_features": f"sandbox_crft.cb_activity_features_{postfix}",
        "cb_potential_features": f"sandbox_crft.cb_potential_features_{postfix}",
        "cb_share_features": f"sandbox_crft.cb_share_features_{postfix}",
        "cb_dataset": f"sandbox_crft.cb_client_month_dataset_{postfix}",
        "loyalty_client_categories": "sandbox_crft.mae_loyalty_client_categories",
    }


def run_sql(conn, queries, compute_stats_table=None, preview_table=None, limit=5):
    """Выполняет набор SQL-запросов и при необходимости показывает preview."""
    with conn:
        for query in queries:
            conn.execute(query)
        if compute_stats_table is not None:
            conn.execute(f"compute stats {compute_stats_table}")
        if preview_table is not None:
            return conn.fetch(f"select * from {preview_table} limit {limit}")
    return None


def category_suffix(category_id):
    """Преобразует id категории в короткий суффикс имени фичи."""
    return str(category_id).replace("-", "")[-4:]


def add_category_suffix(categories_df, id_col="title_id"):
    """Добавляет колонку suffix для генерации wide-фичей по категориям."""
    df = categories_df.copy()
    df["suffix"] = df[id_col].apply(category_suffix)
    return df


def generate_category_spend_sql(categories_df):
    """Генерирует SQL-агрегации трат и количества операций по категориям."""
    parts = []
    for _, row in categories_df.iterrows():
        category_id = row["title_id"]
        suffix = row["suffix"]
        parts.append(f"""
        sum(case
                when t.title_id = '{category_id}'
                 and t.month_date = add_months(b.forecast_month, -2)
                then coalesce(t.charge, 0) else 0
            end) as spend_cat_{suffix}_lag1m,
        sum(case
                when t.title_id = '{category_id}'
                 and t.month_date = add_months(b.forecast_month, -3)
                then coalesce(t.charge, 0) else 0
            end) as spend_cat_{suffix}_lag2m,
        sum(case
                when t.title_id = '{category_id}'
                 and t.month_date = add_months(b.forecast_month, -4)
                then coalesce(t.charge, 0) else 0
            end) as spend_cat_{suffix}_lag3m,
        sum(case
                when t.title_id = '{category_id}'
                 and t.month_date >= add_months(b.forecast_month, -4)
                 and t.month_date < add_months(b.forecast_month, -1)
                then coalesce(t.charge, 0) else 0
            end) as spend_cat_{suffix}_3m,
        sum(case
                when t.title_id = '{category_id}'
                 and t.month_date >= add_months(b.forecast_month, -7)
                 and t.month_date < add_months(b.forecast_month, -1)
                then coalesce(t.charge, 0) else 0
            end) as spend_cat_{suffix}_6m,
        sum(case
                when t.title_id = '{category_id}'
                 and t.month_date >= add_months(b.forecast_month, -4)
                 and t.month_date < add_months(b.forecast_month, -1)
                then coalesce(t.count_charge, 0) else 0
            end) as cnt_cat_{suffix}_3m,
        sum(case
                when t.title_id = '{category_id}'
                 and t.month_date >= add_months(b.forecast_month, -7)
                 and t.month_date < add_months(b.forecast_month, -1)
                then coalesce(t.count_charge, 0) else 0
            end) as cnt_cat_{suffix}_6m
        """)
    return ",\n".join(parts)


def generate_offered_percent_sql(categories_df):
    """Генерирует SQL-агрегации предложенных процентов по категориям."""
    parts = []
    for _, row in categories_df.iterrows():
        category_id = row["title_id"]
        suffix = row["suffix"]
        parts.append(f"""
        max(case
                when o.category_id = '{category_id}'
                then cast(regexp_replace(cast(o.category_percent as string), ',', '.') as double)
                else 0
            end) as offered_percent_cat_{suffix}
        """)
    return ",\n".join(parts)


def generate_selected_history_sql(categories_df):
    """Генерирует SQL-агрегации истории выбранных категорий."""
    parts = []
    for _, row in categories_df.iterrows():
        category_id = row["title_id"]
        suffix = row["suffix"]
        parts.append(f"""
        max(case
                when s.category_id = '{category_id}'
                 and trunc(cast(s.end_date as timestamp), 'MM') = add_months(b.forecast_month, -2)
                then 1 else 0
            end) as was_selected_cat_{suffix}_lag1m,
        max(case
                when s.category_id = '{category_id}'
                 and trunc(cast(s.end_date as timestamp), 'MM') >= add_months(b.forecast_month, -4)
                 and trunc(cast(s.end_date as timestamp), 'MM') < add_months(b.forecast_month, -1)
                then 1 else 0
            end) as was_selected_cat_{suffix}_3m,
        max(case
                when s.category_id = '{category_id}'
                 and trunc(cast(s.end_date as timestamp), 'MM') >= add_months(b.forecast_month, -7)
                 and trunc(cast(s.end_date as timestamp), 'MM') < add_months(b.forecast_month, -1)
                then 1 else 0
            end) as was_selected_cat_{suffix}_6m,
        avg(case
                when s.category_id = '{category_id}'
                 and trunc(cast(s.end_date as timestamp), 'MM') >= add_months(b.forecast_month, -7)
                 and trunc(cast(s.end_date as timestamp), 'MM') < add_months(b.forecast_month, -1)
                then cast(regexp_replace(cast(s.category_percent as string), ',', '.') as double)
            end) as avg_selected_percent_cat_{suffix}_6m
        """)
    return ",\n".join(parts)


def generate_share_sql(categories_df):
    """Генерирует SQL-фичи долей трат по категориям."""
    parts = []
    for _, row in categories_df.iterrows():
        suffix = row["suffix"]
        parts.append(f"""
        case
            when coalesce(total_spend_3m, 0) > 0
            then coalesce(spend_cat_{suffix}_3m, 0) / total_spend_3m
            else 0
        end as share_spend_cat_{suffix}_3m
        """)
    return ",\n".join(parts)


def pyarrow_to_pandas(path):
    """Читает parquet через pyarrow с приведением decimal/binary типов."""
    import pyarrow as pa
    import pyarrow.dataset as ds

    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table()
    new_fields = []

    for name, field in zip(table.schema.names, table.schema):
        field_type = field.type
        if pa.types.is_decimal(field_type):
            new_type = pa.float64()
        elif pa.types.is_binary(field_type):
            new_type = pa.string()
        else:
            new_type = field_type
        new_fields.append(pa.field(name, new_type))

    table = table.cast(pa.schema(new_fields))
    return table.to_pandas()


def read_partitioned_parquet_folder(title_path):
    """Читает все parquet-файлы из папки и склеивает их в один датафрейм."""
    title_path = Path(title_path)
    frames = []
    for file_path in sorted(title_path.iterdir()):
        if file_path.is_file() or file_path.is_dir():
            frames.append(pyarrow_to_pandas(str(file_path)))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def optimize_dataframe_types(df):
    """Снижает потребление памяти: float64->float32, int64->минимальный int/uint."""
    result = df.copy()

    for col in result.select_dtypes(include=["float64"]).columns:
        result[col] = result[col].astype("float32")

    for col in result.select_dtypes(include=["int64"]).columns:
        col_min = result[col].min()
        col_max = result[col].max()
        if pd.isna(col_min) or pd.isna(col_max):
            continue
        if col_min >= 0:
            if col_max < np.iinfo(np.uint8).max:
                result[col] = result[col].astype(np.uint8)
            elif col_max < np.iinfo(np.uint16).max:
                result[col] = result[col].astype(np.uint16)
            elif col_max < np.iinfo(np.uint32).max:
                result[col] = result[col].astype(np.uint32)
        else:
            if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                result[col] = result[col].astype(np.int8)
            elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                result[col] = result[col].astype(np.int16)
            elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                result[col] = result[col].astype(np.int32)
    return result


def make_season_series(pos_values):
    series = pd.Series(pos_values, dtype="float64")
    series.index = pd.to_datetime(series.index)
    return series


def get_honest_season_coef(forecast_month, pos_series):
    from nbc_project.training_helpers import get_honest_season_coef as shared_coef
    return shared_coef(forecast_month, pos_values=pos_series)


def add_season_features(df, pos_values, season_base_cols, month_col="forecast_month", clip_min=0.75, clip_max=1.50):
    """Compatibility helper: uses the exact same formula as train."""
    from nbc_project.training_helpers import add_standard_time_features
    return add_standard_time_features(
        df, month_col=month_col, season_base_cols=season_base_cols,
        pos_values=pos_values, clip_min=clip_min, clip_max=clip_max,
    )


def add_calendar_features(df, month_col="forecast_month"):
    """Добавляет календарные признаки месяца прогноза."""
    result = df.copy()
    result[month_col] = pd.to_datetime(result[month_col])
    result["forecast_month_num"] = result[month_col].dt.month
    result["forecast_quarter"] = result[month_col].dt.quarter
    result["is_april"] = (result["forecast_month_num"] == 4).astype(int)
    result["is_march"] = (result["forecast_month_num"] == 3).astype(int)
    result["is_may"] = (result["forecast_month_num"] == 5).astype(int)
    result["is_december"] = (result["forecast_month_num"] == 12).astype(int)
    result["is_q_end"] = result["forecast_month_num"].isin([3, 6, 9, 12]).astype(int)
    result["month_sin"] = np.sin(2 * np.pi * result["forecast_month_num"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["forecast_month_num"] / 12)
    return result


def prepare_inference_frame(
    df,
    pos_values,
    season_base_cols,
    cat_features,
    numeric_cols=None,
    preprocessing_mode="training_aligned",
):
    """Приводит inference frame к формату модели.

    legacy — сохраняет текущую inference-логику проекта.
    training_aligned — использует add_standard_time_features из train.
    """
    result = optimize_dataframe_types(df)

    if preprocessing_mode == "training_aligned":
        from nbc_project.training_helpers import (
            SEASON_BASE_COLS,
            add_standard_time_features,
        )

        cat_set = set(cat_features or [])

        for col in result.columns:
            if col not in cat_set and result[col].dtype == "object":
                converted = pd.to_numeric(result[col], errors="coerce")
                if converted.notna().sum() > 0:
                    result[col] = converted

        # Train использует полный SEASON_BASE_COLS.
        # Config может содержать дополнительные season bases,
        # но не должен заменять train defaults.
        effective_season_base_cols = list(
            dict.fromkeys(
                list(SEASON_BASE_COLS)
                + list(season_base_cols or [])
            )
        )

        result = add_standard_time_features(
            result,
            month_col="forecast_month",
            season_base_cols=effective_season_base_cols,
        )
    elif preprocessing_mode == "legacy":
        result = add_season_features(result, pos_values=pos_values, season_base_cols=season_base_cols)
        result = add_calendar_features(result)
    else:
        raise ValueError(f"Unknown preprocessing_mode={preprocessing_mode}")

    for col in numeric_cols or []:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    existing_cat_features = [col for col in cat_features if col in result.columns]
    if existing_cat_features:
        result[existing_cat_features] = result[existing_cat_features].fillna("missing").astype(str)
    return result

def load_catboost_models(model_paths, project_root=None, only_existing_optional=True):
    """Загружает CatBoost-модели из словаря {model_key: {path, type}}.

    Если у модели стоит optional=true и файла нет, модель пропускается. Это удобно
    для старого inference-ноутбука, где модели cashback/loyalty могли быть
    временно закомментированы.
    """
    from catboost import CatBoostClassifier, CatBoostRegressor

    root = Path(project_root) if project_root is not None else Path(".")
    models = {}
    for model_key, spec in model_paths.items():
        model_type = spec.get("type", "regression")
        path = Path(spec["path"] if isinstance(spec, dict) else spec)
        if not path.is_absolute():
            path = root / path

        if spec.get("optional") and only_existing_optional and not path.exists():
            print(f"Пропускаю optional-модель {model_key}: файл не найден {path}")
            continue

        model = CatBoostClassifier() if model_type == "classification" else CatBoostRegressor()
        model.load_model(str(path))
        models[model_key] = model
    return models


def load_optional_calibrators(model_paths, project_root=None):
    """Загружает калибраторы вероятностей, если они указаны и существуют."""
    import joblib

    root = Path(project_root) if project_root is not None else Path(".")
    calibrators = {}
    for model_key, spec in model_paths.items():
        calibrator_path = spec.get("calibrator_path")
        if not calibrator_path:
            continue
        path = Path(calibrator_path)
        if not path.is_absolute():
            path = root / path
        if path.exists():
            calibrators[model_key] = joblib.load(path)
        else:
            print(f"Калибратор для {model_key} не найден: {path}")
    return calibrators


def model_output_columns(model_paths):
    """Возвращает словарь {model_key: output_col} из конфига моделей."""
    return {key: spec.get("output_col", f"{key}_pred") for key, spec in model_paths.items()}


def predict_with_models(df, models, output_columns=None, calibrators=None, strict_features=False):
    """Скорит датафрейм набором CatBoost-моделей.

    Для классификаторов используется predict_proba[:, 1]. Если передан calibrator,
    дополнительно применяется калибровка вероятности.
    """
    from catboost import CatBoostClassifier, Pool

    result = df.copy()
    output_columns = output_columns or {}
    calibrators = calibrators or {}

    for model_key, model in models.items():
        features = list(model.feature_names_)
        cat_feature_indices = model.get_cat_feature_indices()
        missing = [col for col in features if col not in result.columns]
        if missing and strict_features:
            raise ValueError(f"Для модели {model_key} отсутствуют {len(missing)} features: {missing[:30]}")
        pool = Pool(result.reindex(columns=features), cat_features=cat_feature_indices)

        out_col = output_columns.get(model_key, f"{model_key}_pred")
        if isinstance(model, CatBoostClassifier):
            raw_col = f"{out_col}_raw"
            proba = model.predict_proba(pool)[:, 1]
            if model_key in calibrators:
                result[raw_col] = proba
                result[out_col] = calibrators[model_key].transform(proba)
            else:
                result[out_col] = proba
        else:
            pred = model.predict(pool)
        
            # Для регрессионных моделей не даём отрицательные значения
            pred = np.clip(pred, 0, None)
        
            # Для cashback-регрессий дополнительно режем сверху до 5000
            if "cashback" in model_key.lower() or "cashback" in out_col.lower():
                pred = np.clip(pred, 0, 5000)
        
            result[out_col] = pred
            
    return result


def predict_with_models_debug(df, models, output_columns=None, calibrators=None):
    import numpy as np
    import pandas as pd
    import traceback
    from catboost import CatBoostClassifier, Pool

    result = df.copy()
    output_columns = output_columns or {}
    calibrators = calibrators or {}

    for model_key, model in models.items():
        print("\n" + "=" * 80)
        print(f"SCORING MODEL: {model_key}")
        print("=" * 80)

        features = list(model.feature_names_)
        cat_feature_indices = list(model.get_cat_feature_indices())
        cat_features = [features[i] for i in cat_feature_indices]

        print("features count:", len(features))
        print("cat features count:", len(cat_features))
        print("cat features sample:", cat_features[:30])

        if len(features) > 138:
            print("feature_idx=138:", features[138])
            bad_col = features[138]
            if bad_col in result.columns:
                print("feature_idx=138 dtype:", result[bad_col].dtype)
                print(result[bad_col].value_counts(dropna=False).head(20))
            else:
                print("feature_idx=138 отсутствует в result")

        missing_features = [c for c in features if c not in result.columns]
        print("missing features:", len(missing_features))
        if missing_features:
            print(missing_features[:50])

        X = result.reindex(columns=features).copy()

        object_cols = [c for c in X.columns if X[c].dtype == "object"]
        object_not_cat = [c for c in object_cols if c not in cat_features]

        print("object cols:", len(object_cols))
        print("object cols NOT cat:", len(object_not_cat))
        if object_not_cat:
            print("object_not_cat sample:", object_not_cat[:50])

            for c in object_not_cat[:10]:
                print("\nBAD OBJECT NUMERIC COL:", c)
                print("dtype:", X[c].dtype)
                print(X[c].value_counts(dropna=False).head(10))

        try:
            print("Creating Pool...")
            pool = Pool(
                X,
                cat_features=cat_feature_indices,
            )
            print("Pool OK")

            out_col = output_columns.get(model_key, f"{model_key}_pred")

            if isinstance(model, CatBoostClassifier):
                print("Predicting classifier...")
                raw_col = f"{out_col}_raw"
                proba = model.predict_proba(pool)[:, 1]

                result[raw_col] = proba

                if model_key in calibrators:
                    print("Applying calibrator...")
                    result[out_col] = calibrators[model_key].transform(proba)
                else:
                    result[out_col] = proba

            else:
                print("Predicting regressor...")
                pred = model.predict(pool)

                pred = np.clip(pred, 0, None)

                if "cashback" in model_key.lower() or "cashback" in out_col.lower():
                    pred = np.clip(pred, 0, 5000)

                result[out_col] = pred

            print(f"MODEL OK: {model_key}")

        except Exception as e:
            print("\nMODEL FAILED:", model_key)
            print("ERROR:", repr(e))

            print("\nTraceback:")
            traceback.print_exc()

            print("\nMost likely bad object_not_cat columns:")
            print(object_not_cat[:100])

            raise

    return result


def upload_folder_to_hdfs(source_folder, hdfs_path, namenode_url):
    """Загружает все csv из локальной папки в HDFS, предварительно очищая папку."""
    from hdfs.ext.kerberos import KerberosClient

    source_folder = Path(source_folder)
    client = KerberosClient(namenode_url)
    files = client.list(hdfs_path)
    print(f"files in hdfs dir = {files}")

    for file_name in files:
        client.delete(os.path.join(hdfs_path, file_name))
        print(f"Deleted data in HDFS: {file_name}")

    print(f"Start copying data to HDFS: {hdfs_path}")
    for csv_file in source_folder.glob("*.csv"):
        full_hdfs_path = os.path.join(hdfs_path, csv_file.name)
        with open(csv_file, mode="rb") as f:
            client.write(full_hdfs_path, data=f)
        client.set_permission(full_hdfs_path, "0777")
        print(f"File {csv_file.name} written successfully; permissions changed to 0777")


def _sql_literal(value):
    """Безопасно приводит простое значение к SQL-литералу для шаблонов."""
    if value is None:
        return "null"
    return "'" + str(value).replace("'", "''") + "'"


def generate_send_to_dbo_scenario_mart_sql(
    send_to_dbo,
    send_to_dbo_base,
    gusev_table,
    theme_default,
    target_month=None,
    extra_sources=None,
    historical_union_table=None,
    historical_union_month=None,
):
    """
    Генерирует SQL для send_to_dbo как long-витрины сценариев предложений.

    Итоговая схема сохраняет старые обязательные поля:
    - cdi_id
    - period_start_date
    - category_id
    - category_percent

    И добавляет поля аудита сценария:
    - offer_scenario_id: сценарий/источник предложения;
    - offer_source: человекочитаемый источник;
    - source_priority: приоритет источника;
    - source_score: скор внутри источника, если есть;
    - source_rank: ранг внутри источника;
    - final_rank: финальный ранг внутри клиента, месяца и сценария;
    - is_random_fallback: флаг случайного добора.

    extra_sources — список словарей из configs/inference_config.json.
    Каждый источник должен быть уже в long-формате или приводиться к нему:
    client_id_col, period_col, category_col, percent_col, score_col, table.
    """
    extra_sources = extra_sources or []

    historical_sql = ""
    if historical_union_table and historical_union_month:
        historical_sql = f"""
    union all
    select
        cast(cdi_id as bigint) as cdi_id,
        cast(period_start_date as timestamp) as period_start_date,
        cast(category_id as string) as category_id,
        cast(regexp_replace(cast(category_percent as string), ',', '.') as double) as category_percent,
        'historical' as offer_scenario_id,
        'historical' as offer_source,
        99 as source_priority,
        cast(null as double) as source_score,
        cast(null as int) as source_rank,
        cast(null as int) as final_rank,
        0 as is_random_fallback
    from {historical_union_table}
    where period_start_date = '{historical_union_month}'
"""

    extra_sql_parts = []
    for idx, source in enumerate(extra_sources, start=1):
        if not source.get("enabled", False):
            continue

        table = source["table"]
        scenario_id = source.get("scenario_id", source.get("source", f"source_{idx}"))
        source_name = source.get("source", scenario_id)
        priority = int(source.get("priority", idx + 1))
        client_col = source.get("client_id_col", "cdi_id")
        period_col = source.get("period_col", "period_start_date")
        category_col = source.get("category_col", "category_id")
        percent_col = source.get("percent_col", "category_percent")
        score_col = source.get("score_col")
        rank_col = source.get("rank_col")
        where_sql = source.get("where_sql", "1=1")

        if target_month:
            where_sql = f"({where_sql}) and cast({period_col} as timestamp) = cast('{target_month}' as timestamp)"

        source_score_sql = f"cast({score_col} as double)" if score_col else "cast(null as double)"
        source_rank_sql = f"cast({rank_col} as int)" if rank_col else "cast(null as int)"

        extra_sql_parts.append(f"""
    union all
    select
        cast({client_col} as bigint) as cdi_id,
        cast({period_col} as timestamp) as period_start_date,
        cast({category_col} as string) as category_id,
        cast(regexp_replace(cast({percent_col} as string), ',', '.') as double) as category_percent,
        {_sql_literal(scenario_id)} as offer_scenario_id,
        {_sql_literal(source_name)} as offer_source,
        {priority} as source_priority,
        {source_score_sql} as source_score,
        {source_rank_sql} as source_rank,
        cast(null as int) as final_rank,
        case when lower({_sql_literal(source_name)}) = 'random' then 1 else 0 end as is_random_fallback
    from {table}
    where {where_sql}
""")

    extra_sql = "".join(extra_sql_parts)

    return f"""
drop table if exists {send_to_dbo};

create table {send_to_dbo} stored as parquet as
with gusev_selected as (
    select
        cast(b.party_id as bigint) as cdi_id,
        g.theme_id,
        row_number() over (
            partition by cast(b.party_id as bigint)
            order by cast(g.assign_date as timestamp) desc
        ) as rn_gusev
    from {gusev_table} g
    join acrm.o_client b
        on g.client_id = b.id
),
client_theme as (
    select
        s.cdi_id,
        coalesce(g.theme_id, '{theme_default}') as final_theme_id,
        case
            when g.cdi_id is not null then 1
            else 0
        end as selected_from_gusev_flg
    from (
        select distinct
            cast(cdi_id as bigint) as cdi_id
        from {send_to_dbo_base}
    ) s
    left join gusev_selected g
        on s.cdi_id = g.cdi_id
       and g.rn_gusev = 1
),
base_theme as (
    select
        cast(s.cdi_id as bigint) as cdi_id,
        cast(s.period_start_date as timestamp) as period_start_date,
        cast(s.category_id as string) as category_id,
        cast(regexp_replace(cast(s.category_percent as string), ',', '.') as double) as category_percent,
        'base_theme' as offer_scenario_id,
        case
            when ct.selected_from_gusev_flg = 1 then 'gusev_theme'
            else 'default_theme'
        end as offer_source,
        1 as source_priority,
        cast(null as double) as source_score,
        cast(null as int) as source_rank,
        cast(null as int) as final_rank,
        0 as is_random_fallback
    from {send_to_dbo_base} s
    join client_theme ct
        on cast(s.cdi_id as bigint) = ct.cdi_id
       and s.theme_id = ct.final_theme_id
),
all_scenarios_raw as (
    select * from base_theme
    {historical_sql}
    {extra_sql}
),
deduplicated as (
    select *
    from (
        select
            cdi_id,
            period_start_date,
            category_id,
            case
                when category_id in ('1133', '1134') then least(category_percent, 5)
                else category_percent
            end as category_percent,
            offer_scenario_id,
            offer_source,
            source_priority,
            source_score,
            source_rank,
            is_random_fallback,
            row_number() over (
                partition by cdi_id, period_start_date, offer_scenario_id, category_id
                order by source_priority asc, case when source_score is null then 1 else 0 end asc, source_score desc
            ) as rn_dup
        from all_scenarios_raw
    ) t
    where rn_dup = 1
),
ranked as (
    select
        cdi_id,
        period_start_date,
        category_id,
        category_percent,
        offer_scenario_id,
        offer_source,
        source_priority,
        source_score,
        source_rank,
        row_number() over (
            partition by cdi_id, period_start_date, offer_scenario_id
            order by source_priority asc, case when source_score is null then 1 else 0 end asc, source_score desc, category_id
        ) as final_rank,
        is_random_fallback
    from deduplicated
)
select
    cdi_id,
    period_start_date,
    category_id,
    category_percent,
    offer_scenario_id,
    offer_source,
    source_priority,
    source_score,
    source_rank,
    final_rank,
    is_random_fallback
from ranked;
"""


def add_offer_scenario_filter(table_alias, filter_sql=None):
    """Возвращает SQL-условие для выбора активного сценария из send_to_dbo."""
    if not filter_sql:
        return ""
    return f"\n   and ({filter_sql.format(alias=table_alias)})"


def infer_required_input_columns(models, config=None, id_columns=None):
    """Возвращает минимальный набор колонок, который надо читать из parquet для inference.

    Берём union feature_names_ всех CatBoost-моделей, но исключаем фичи,
    которые создаются внутри prepare_inference_frame(): calendar/season фичи.
    Дополнительно добавляем id/date колонки и season_base_cols из конфига.
    """
    config = config or {}
    id_columns = list(id_columns or ["cdi_id", "client_id", "forecast_month", "period_start_date"])

    generated_cols = {
        "season_total_pos_coef_honest",
        "forecast_month_num",
        "forecast_quarter",
        "is_april",
        "is_march",
        "is_may",
        "is_december",
        "is_q_end",
        "month_sin",
        "month_cos",
    }

    features = set()
    for model in models.values():
        features.update(getattr(model, "feature_names_", []) or [])

    season_base_cols = set(config.get("season_base_cols", []))
    numeric_string_cols = set(config.get("numeric_string_cols", []))
    cat_features = set(config.get("cat_features", []))
    score_columns = set(config.get("score_columns", []))

    # Поддерживаем и legacy `seasonadj`, и train-aligned `_season_adj`.
    season_adj_cols = {
        col for col in features
        if str(col).endswith("seasonadj") or str(col).endswith("_season_adj")
    }
    derived_season_bases = set()
    for col in season_adj_cols:
        text = str(col)
        if text.endswith("_season_adj"):
            derived_season_bases.add(text[:-len("_season_adj")])
        elif text.endswith("seasonadj"):
            derived_season_bases.add(text[:-len("seasonadj")])

    read_cols = (
        features
        | season_base_cols
        | derived_season_bases
        | numeric_string_cols
        | cat_features
        | score_columns
        | set(id_columns)
    )
    read_cols = read_cols - generated_cols - season_adj_cols

    return sorted(read_cols)


# def iter_parquet_dataset_batches(input_path, columns=None, batch_size=100_000):
#     """Итерирует parquet-папку батчами, не собирая весь dataset в pandas.

#     Возвращает pandas.DataFrame для каждого batch. Колонки, отсутствующие в parquet,
#     тихо выкидываются из columns, чтобы можно было передавать union фичей моделей.
#     """
#     import gc
#     import pyarrow as pa
#     import pyarrow.dataset as ds

#     dataset = ds.dataset(str(input_path), format="parquet")
#     schema_names = set(dataset.schema.names)

#     if columns is not None:
#         columns = [col for col in columns if col in schema_names]

#     scanner = dataset.scanner(columns=columns, batch_size=batch_size)

#     for batch in scanner.to_batches():
#         table = pa.Table.from_batches([batch])

#         new_fields = []
#         for name, field in zip(table.schema.names, table.schema):
#             field_type = field.type
#             if pa.types.is_decimal(field_type):
#                 new_type = pa.float64()
#             elif pa.types.is_binary(field_type):
#                 new_type = pa.string()
#             else:
#                 new_type = field_type
#             new_fields.append(pa.field(name, new_type))

#         table = table.cast(pa.schema(new_fields))
#         df_part = table.to_pandas()
#         yield df_part

#         del df_part, table, batch
#         gc.collect()


def iter_parquet_dataset_batches(
    input_path,
    columns=None,
    batch_size=10_000,
):
    """Читает parquet dataset небольшими batch без readahead."""

    import gc
    import pyarrow as pa
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(input_path), format="parquet")
    schema_names = set(dataset.schema.names)

    if columns is not None:
        columns = [
            column
            for column in columns
            if column in schema_names
        ]

    scanner = dataset.scanner(
        columns=columns,
        batch_size=batch_size,
        use_threads=False,
        batch_readahead=1,
        fragment_readahead=1,
    )

    for batch in scanner.to_batches():
        table = pa.Table.from_batches([batch])

        # RecordBatch больше не нужен — Table уже создана.
        del batch

        new_fields = []
        schema_changed = False

        for field in table.schema:
            old_type = field.type
            new_type = old_type

            if pa.types.is_decimal(old_type):
                new_type = pa.float64()
            elif pa.types.is_binary(old_type):
                new_type = pa.string()

            if new_type != old_type:
                schema_changed = True

            new_fields.append(
                pa.field(
                    field.name,
                    new_type,
                    nullable=field.nullable,
                )
            )

        # Не делаем полный cast без необходимости.
        if schema_changed:
            table = table.cast(pa.schema(new_fields))

        df_part = table.to_pandas(
            split_blocks=True,
            self_destruct=True,
        )

        # Arrow-объект освобождаем до передачи batch в скоринг.
        del table
        gc.collect()

        try:
            yield df_part
        finally:
            del df_part
            gc.collect()



def score_inference_parquet_in_batches(
    input_path,
    output_dir,
    models,
    inference_config,
    output_columns=None,
    calibrators=None,
    columns=None,
    batch_size=100_000,
    score_columns=None,
    offer_batch_func=None,
    file_format="csv",
):
    """Скорит inference parquet-папку батчами и сразу пишет результаты на диск.

    В памяти одновременно находится только один batch.

    Parameters
    ----------
    input_path : str | Path
        Parquet-папка inference dataset.
    output_dir : str | Path
        Папка для part-файлов scores_part_XXXX.csv/parquet.
    models : dict
        Загруженные CatBoost-модели.
    inference_config : dict
        configs/inference_config.json.
    output_columns, calibrators : dict
        Как в predict_with_models().
    columns : list[str] | None
        Какие колонки читать из parquet. Если None — вычисляется автоматически.
    offer_batch_func : callable | None
        Функция, которая применяет генерацию офферов к одному batch:
        df_part = offer_batch_func(df_part, batch_num).
    file_format : {'csv', 'parquet'}
        Формат part-файлов. Для дальнейшей загрузки через текущий HDFS-код удобнее csv.
    """
    import gc
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_columns = output_columns or {}
    calibrators = calibrators or {}

    if columns is None:
        columns = infer_required_input_columns(models, config=inference_config)

    score_columns = list(score_columns or inference_config.get("score_columns", []))

    paths = []
    rows_total = 0

    for batch_num, df_part in enumerate(
        iter_parquet_dataset_batches(input_path, columns=columns, batch_size=batch_size),
        start=1,
    ):
        if df_part.empty:
            continue

        if offer_batch_func is not None:
            df_part = offer_batch_func(df_part, batch_num)

        df_part = prepare_inference_frame(
            df_part,
            pos_values=inference_config.get("pos_values", {}),
            season_base_cols=inference_config.get("season_base_cols", []),
            cat_features=inference_config.get("cat_features", []),
            numeric_cols=inference_config.get("numeric_string_cols", []),
            preprocessing_mode=inference_config.get("preprocessing_mode", "training_aligned"),
        )

        df_scored = predict_with_models(
            df_part,
            models=models,
            output_columns=output_columns,
            calibrators=calibrators,
            strict_features=inference_config.get("strict_features", True),
        )

        # df_scored = predict_with_models_debug(
        #     df_part,
        #     models=models,
        #     output_columns=output_columns,
        #     calibrators=calibrators,
        # )

        if {"cashback_pred", "loyal_pred"}.issubset(df_scored.columns):
            df_scored["cashback_expect"] = df_scored["cashback_pred"] * df_scored["loyal_pred"]
            df_scored["pos_expect"] = df_scored["pos_loyal_pred"] * df_scored["loyal_pred"] + df_scored["pos_noloyal_pred"] * (1 - df_scored["loyal_pred"])

        existing_score_columns = [col for col in score_columns if col in df_scored.columns]
        if not existing_score_columns:
            # fallback: сохраняем id/date + все *_pred.
            existing_score_columns = [
                col for col in ["cdi_id", "client_id", "forecast_month", "period_start_date"]
                if col in df_scored.columns
            ] + [col for col in df_scored.columns if col.endswith("_pred")]

        out = df_scored[existing_score_columns].copy()
        out["batch_num"] = batch_num

        if file_format == "parquet":
            path = output_dir / f"scores_part_{batch_num:04d}.parquet"
            out.to_parquet(path, index=False)
        elif file_format == "csv":
            path = output_dir / f"scores_part_{batch_num:04d}.csv"
            out.to_csv(path, index=False)
        else:
            raise ValueError("file_format должен быть 'csv' или 'parquet'")

        paths.append(path)
        rows_total += len(out)
        print(f"batch={batch_num}, rows={len(out):,}, rows_total={rows_total:,}, saved={path}")

        del df_part, df_scored, out
        gc.collect()

    print(f"Готово. Всего строк: {rows_total:,}. Файлов: {len(paths)}. Папка: {output_dir}")
    return paths

