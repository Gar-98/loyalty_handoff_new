
"""Повторяющиеся функции для ноутбука `1. data_fm (1).ipynb`.

Здесь нет бизнес-логики: только технические утилиты для чтения parquet,
оптимизации типов, сборки SQL-списков фичей и создания feature-selected таблиц.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def cast_arrow_table(table):
    """Приводит проблемные pyarrow-типы к типам, которые нормально читаются в pandas."""
    import pyarrow as pa

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

    return table.cast(pa.schema(new_fields))


def pyarrow_to_pandas(path, filters=None):
    """Читает parquet-папку через pyarrow и возвращает pandas DataFrame."""
    import pyarrow.dataset as ds

    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(filter=filters)
    table = cast_arrow_table(table)
    return table.to_pandas()


def pyarrow_random_sample_to_pandas(path, sample_frac=0.1, max_rows=None, batch_size=50_000, random_state=42):
    """Читает parquet батчами и берёт воспроизводимую случайную подвыборку."""
    import pyarrow.dataset as ds

    rng = np.random.default_rng(random_state)
    dataset = ds.dataset(path, format="parquet")

    parts = []
    rows_collected = 0
    for batch in dataset.to_batches(batch_size=batch_size):
        table = cast_arrow_table(batch.to_table())
        df_batch = table.to_pandas()

        if len(df_batch) == 0:
            continue

        mask = rng.random(len(df_batch)) < sample_frac
        sampled = df_batch.loc[mask]
        if len(sampled) == 0:
            continue

        parts.append(sampled)
        rows_collected += len(sampled)

        if max_rows is not None and rows_collected >= max_rows:
            break

    if not parts:
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    if max_rows is not None and len(result) > max_rows:
        result = result.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
    return result


def drop_existing_columns(df, columns, keep_columns=None):
    """Удаляет только те колонки, которые реально есть в датафрейме."""
    keep_columns = set(keep_columns or [])
    drop_columns = [col for col in columns if col in df.columns and col not in keep_columns]
    return df.drop(columns=drop_columns)


def optimize_dataframe_dtypes(df):
    """Уменьшает потребление памяти: float64 -> float32, int64 -> минимально подходящий int."""
    result = df.copy()

    for col in result.select_dtypes(include=["float64"]).columns:
        result[col] = result[col].astype("float32")

    for col in result.select_dtypes(include=["int64"]).columns:
        col_min = result[col].min()
        col_max = result[col].max()

        if pd.isna(col_min) or pd.isna(col_max):
            continue

        if col_min >= 0:
            if col_max < 255:
                result[col] = result[col].astype(np.uint8)
            elif col_max < 65_535:
                result[col] = result[col].astype(np.uint16)
            elif col_max < 4_294_967_295:
                result[col] = result[col].astype(np.uint32)
        else:
            if col_min > -128 and col_max < 127:
                result[col] = result[col].astype(np.int8)
            elif col_min > -32_768 and col_max < 32_767:
                result[col] = result[col].astype(np.int16)
            elif col_min > -2_147_483_648 and col_max < 2_147_483_647:
                result[col] = result[col].astype(np.int32)

    return result


def prepare_dataframe_for_feature_selection(df, drop_columns=None, keep_columns=None):
    """Удаляет технические колонки и оптимизирует типы перед сохранением CSV."""
    result = df.copy()
    if drop_columns is not None:
        result = drop_existing_columns(result, drop_columns, keep_columns=keep_columns)
    return optimize_dataframe_dtypes(result)


def save_optimized_csv(df, path, drop_columns=None, keep_columns=None, index=False):
    """Готовит датафрейм к feature selection и сохраняет в CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = prepare_dataframe_for_feature_selection(df, drop_columns=drop_columns, keep_columns=keep_columns)
    result.to_csv(path, index=index)
    return result


def load_optimize_save_dataset(input_path, output_csv_path, drop_columns=None, keep_columns=None):
    """Читает parquet, оптимизирует типы, сохраняет CSV и возвращает датафрейм."""
    df = pyarrow_to_pandas(input_path)
    return save_optimized_csv(df, output_csv_path, drop_columns=drop_columns, keep_columns=keep_columns)


def unique_features(*feature_lists):
    """Склеивает списки фичей с сохранением порядка и удалением дублей."""
    result = []
    seen = set()
    for feature_list in feature_lists:
        for feature in feature_list:
            if feature not in seen:
                result.append(feature)
                seen.add(feature)
    return result


def sql_columns(features):
    """Преобразует список колонок в SQL-фрагмент для SELECT."""
    return ", ".join(features)


def create_selected_features_table(conn, output_table, source_table, features, forecast_month_col="forecast_month"):
    """Создаёт Hive/Impala-таблицу с выбранными фичами."""
    selected_columns = sql_columns(features)
    query = (
        f"CREATE TABLE {output_table}\n"
        f"STORED AS PARQUET\n"
        f"TBLPROPERTIES('transactional' = 'false')\n"
        f"AS (\n"
        f"    SELECT\n"
        f"        cast({forecast_month_col} as date) as {forecast_month_col},\n"
        f"        {selected_columns}\n"
        f"    FROM {source_table}\n"
        f")\n"
    )

    with conn:
        conn.execute(f"drop table if exists {output_table}")
        conn.execute(query)

    return query
