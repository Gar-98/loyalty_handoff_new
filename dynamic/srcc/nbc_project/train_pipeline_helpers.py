"""
Общие функции для 2.1.train_pipeline.ipynb.
"""

import gc
import os
import subprocess
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def cast_decimal_binary(table):
    """Приводит decimal к float64, binary к string, чтобы pandas не падал на типах pyarrow."""
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

    schema = pa.schema(new_fields)
    return table.cast(schema)


def read_parquet_filtered(path, filter_expr=None, columns=None):
    """Читает parquet-файл/папку с опциональным pyarrow-фильтром и списком колонок."""
    import pyarrow.dataset as ds

    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(filter=filter_expr, columns=columns)
    table = cast_decimal_binary(table)
    return table.to_pandas()


def read_parquet_folder(title_path, filter_expr=None, columns=None):
    """Читает все parquet-файлы из папки и склеивает в один DataFrame."""
    parts = []
    files = [os.path.join(title_path, file_name) for file_name in os.listdir(title_path)]

    for file_path in files:
        d = read_parquet_filtered(file_path, filter_expr=filter_expr, columns=columns)
        parts.append(d)
        del d
        gc.collect()

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)


# Оставлены старые имена функций, чтобы не ломать ячейки ноутбука.
def get_df(title_path):
    """Совместимость со старым ноутбуком: чтение parquet-папки."""
    return read_parquet_folder(title_path)


# Shared source and seasonal formula, identical to training_workflow / inference.
from .training_helpers import POS_VALUES, get_honest_season_coef, add_standard_time_features


def add_honest_season_coef(df, month_col="forecast_month", out_col="season_total_pos_coef_honest", clip_min=0.75, clip_max=1.50):
    result = add_standard_time_features(df, month_col=month_col, season_base_cols=[],
                                        clip_min=clip_min, clip_max=clip_max)
    if out_col != "season_total_pos_coef_honest":
        result = result.rename(columns={"season_total_pos_coef_honest": out_col})
    return result


def wape(y_true, y_pred):
    """Weighted absolute percentage error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return np.sum(np.abs(y_true - y_pred)) / denom


def smape(y_true, y_pred):
    """Symmetric mean absolute percentage error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom > 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask])


def regression_metrics_table(y_true, y_pred):
    """Возвращает MAE/RMSE/WAPE/sMAPE в формате DataFrame."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return pd.DataFrame({
        "metric": ["MAE", "RMSE", "WAPE", "sMAPE"],
        "value": [mae, rmse, wape(y_true, y_pred), smape(y_true, y_pred)],
    })


def get_df_sampled(title_path, target_col="target_loyalty_connected_flg", negative_ratio=3, random_state=42, columns=None):
    """Читает parquet-папку и семплирует отрицательный класс относительно положительного."""
    import pyarrow.dataset as ds

    loyal_parts = []
    nonloyal_parts = []

    files = [os.path.join(title_path, file_name) for file_name in os.listdir(title_path)]

    for file_path in files:
        try:
            d_pos = read_parquet_filtered(
                file_path,
                filter_expr=(ds.field(target_col) == 1),
                columns=columns,
            )
            if len(d_pos) > 0:
                loyal_parts.append(d_pos)
            del d_pos
            gc.collect()
        except Exception as e:
            print(f"Skip positive file {file_path}: {e}")

    df_pos = pd.concat(loyal_parts, ignore_index=True) if loyal_parts else pd.DataFrame()
    n_pos = len(df_pos)
    n_neg_need = n_pos * negative_ratio

    print(f"Positive rows: {n_pos}")
    print(f"Need negative rows: {n_neg_need}")

    if n_pos == 0:
        raise ValueError("Нет лояльных строк. Проверь target_col или фильтр.")

    per_file_limit = max(int(n_neg_need / max(len(files), 1) * 2), 1000)

    for file_path in files:
        try:
            d_neg = read_parquet_filtered(
                file_path,
                filter_expr=(ds.field(target_col) == 0),
                columns=columns,
            )
            if len(d_neg) > 0:
                take_n = min(len(d_neg), per_file_limit)
                d_neg_sample = d_neg.sample(n=take_n, random_state=random_state)
                nonloyal_parts.append(d_neg_sample)
            del d_neg
            gc.collect()
        except Exception as e:
            print(f"Skip negative file {file_path}: {e}")

    df_neg = pd.concat(nonloyal_parts, ignore_index=True) if nonloyal_parts else pd.DataFrame()

    if len(df_neg) > n_neg_need:
        df_neg = df_neg.sample(n=n_neg_need, random_state=random_state)

    df = pd.concat([df_pos, df_neg], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print("Final shape:", df.shape)
    print(df[target_col].value_counts(dropna=False))
    return df


def loading_at_file(source_folder, hdfs_url="http://sgo-hd044.go.rshbank.ru:9870;http://sgo-hd045.go.rshbank.ru:9870", hdfs_path="/warehouse/tablespace/external/hive/stg_datamarts.db/temp_uploads"):
    """Загружает csv-файлы из локальной папки в HDFS temp_uploads."""
    from hdfs.ext.kerberos import KerberosClient

    client = KerberosClient(hdfs_url)
    files = client.list(hdfs_path)
    print(f"files in hdfs dir = {files}")

    if files:
        for file in files:
            file_path = os.path.join(hdfs_path, file)
            client.delete(file_path)
        print("Deleted data in HDFS")

    print(f"Start copying data to HDFS: {hdfs_path}")
    for csv_file in os.listdir(source_folder):
        if csv_file.split(".")[-1] == "csv":
            csv_file_path = os.path.join(source_folder, csv_file)
            full_hdfs_path = os.path.join(hdfs_path, csv_file)
            with open(csv_file_path, mode="rb") as f:
                client.write(full_hdfs_path, data=f)
            print(f"File {csv_file} written successfully")
            client.set_permission(full_hdfs_path, "0777")
            print("Permissions changed to 0777")
