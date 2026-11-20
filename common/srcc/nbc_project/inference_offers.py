"""Функции для сборки предложенных категорий для инференса.

В проекте модели ждут wide-фичи вида offered_percent_cat_<category_id>.
Этот модуль помогает собрать такие фичи двумя способами:
1) по top-uplift из внешнего проекта;
2) случайно, если нужно сделать синтетический набор предложенных категорий.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def _normalize_category_id(value):
    """Приводит id категории к строковому суффиксу без префикса фичи."""
    value = str(value)
    return value.replace("offered_percent_cat_", "")


def offered_feature_to_category(feature_name):
    """Достаёт id категории из имени фичи offered_percent_cat_<id>."""
    return _normalize_category_id(feature_name)


def offered_category_to_feature(category_id):
    """Собирает имя фичи offered_percent_cat_<id> по id категории."""
    return "offered_percent_cat_" + _normalize_category_id(category_id)


def get_offered_feature_columns(feature_names):
    """Оставляет только фичи предложенных процентов."""
    return [col for col in feature_names if str(col).startswith("offered_percent_cat_")]


def build_offers_from_uplift(
    uplift_df,
    client_col="client_id",
    category_col="category_id",
    uplift_col="uplift_score",
    percent_col=None,
    top_n=9,
    fixed_5_count=8,
    variable_percent_values=(5.0, 7.0, 9.0, 11.0, 13.0, 15.0),
    capped_categories=("1133", "1134"),
    random_seed=42,
):
    """
    Собирает предложенные категории по top-uplift для каждого клиента.

    Первые fixed_5_count категорий получают 5%, оставшиеся получают случайный
    процент из variable_percent_values. Категории из capped_categories не могут
    получить больше 5%.
    """
    rng = np.random.default_rng(random_seed)
    df = uplift_df.copy()
    df[category_col] = df[category_col].astype(str)
    capped_categories = set(map(str, capped_categories))

    required_cols = {client_col, category_col, uplift_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Нет обязательных колонок: {sorted(missing_cols)}")

    df = df.dropna(subset=[client_col, category_col, uplift_col])
    df = (
        df.sort_values([client_col, uplift_col], ascending=[True, False])
          .drop_duplicates([client_col, category_col], keep="first")
    )
    df["rank"] = df.groupby(client_col).cumcount() + 1
    offers = df[df["rank"] <= top_n].copy()

    if percent_col is not None and percent_col in offers.columns:
        offers["offered_percent"] = offers[percent_col].astype(float)
    else:
        offers["offered_percent"] = 5.0
        variable_mask = offers["rank"] > fixed_5_count
        offers.loc[variable_mask, "offered_percent"] = rng.choice(
            list(variable_percent_values),
            size=int(variable_mask.sum()),
            replace=True,
        )

    offers.loc[offers[category_col].isin(capped_categories), "offered_percent"] = (
        offers.loc[offers[category_col].isin(capped_categories), "offered_percent"].clip(upper=5.0)
    )
    offers["feature_name"] = "offered_percent_cat_" + offers[category_col].astype(str)

    wide_df = (
        offers.pivot_table(
            index=client_col,
            columns="feature_name",
            values="offered_percent",
            aggfunc="max",
            fill_value=0.0,
        )
        .reset_index()
    )
    wide_df.columns.name = None
    audit_df = offers[[client_col, category_col, uplift_col, "rank", "offered_percent", "feature_name"]].copy()
    return wide_df, audit_df


def build_random_offers(
    base_df,
    offered_feature_columns,
    client_col="cdi_id",
    month_col="forecast_month",
    fixed_5_count=8,
    variable_count=1,
    variable_percent_values=(5.0, 7.0, 9.0, 11.0, 13.0, 15.0),
    capped_categories=("1133", "1134"),
    random_seed=42,
):
    """
    Случайно выбирает предложенные категории для каждой строки inference.

    Правило:
    - fixed_5_count категорий получают 5%;
    - variable_count категорий получают случайный процент из variable_percent_values;
    - категории capped_categories всегда ограничиваются сверху 5%.

    Возвращает:
    - wide_df с колонками offered_percent_cat_*;
    - audit_df в long-формате для проверки назначений.
    """
    if client_col not in base_df.columns:
        raise ValueError(f"В base_df нет client_col={client_col!r}")

    offered_feature_columns = get_offered_feature_columns(offered_feature_columns)
    if not offered_feature_columns:
        raise ValueError("Не переданы фичи offered_percent_cat_*")

    total_count = int(fixed_5_count) + int(variable_count)
    if len(offered_feature_columns) < total_count:
        raise ValueError(
            f"Для генерации нужно минимум {total_count} категорий, "
            f"а передано только {len(offered_feature_columns)}"
        )

    rng = np.random.default_rng(random_seed)
    capped_categories = set(map(str, capped_categories))
    categories = np.array([offered_feature_to_category(col) for col in offered_feature_columns], dtype=object)

    id_cols = [client_col]
    if month_col is not None and month_col in base_df.columns:
        id_cols.append(month_col)

    rows = base_df[id_cols].drop_duplicates().reset_index(drop=True).copy()
    audit_parts = []

    for row_idx, row in rows.iterrows():
        chosen_categories = rng.choice(categories, size=total_count, replace=False)
        fixed_categories = chosen_categories[:int(fixed_5_count)]
        variable_categories = chosen_categories[int(fixed_5_count):]

        row_offers = []
        for rank, category_id in enumerate(fixed_categories, start=1):
            row_offers.append((rank, category_id, 5.0, "fixed_5"))

        for offset, category_id in enumerate(variable_categories, start=1):
            percent = float(rng.choice(list(variable_percent_values)))
            rank = int(fixed_5_count) + offset
            row_offers.append((rank, category_id, percent, "variable"))

        audit_row = pd.DataFrame(row_offers, columns=["rank", "category_id", "offered_percent", "offer_type"])
        for col in id_cols:
            audit_row[col] = row[col]
        audit_parts.append(audit_row)

    audit_df = pd.concat(audit_parts, ignore_index=True) if audit_parts else pd.DataFrame()

    if audit_df.empty:
        wide_df = rows.copy()
        for col in offered_feature_columns:
            wide_df[col] = 0.0
        return wide_df, audit_df

    audit_df["category_id"] = audit_df["category_id"].astype(str)
    cap_mask = audit_df["category_id"].isin(capped_categories)
    audit_df.loc[cap_mask, "offered_percent"] = audit_df.loc[cap_mask, "offered_percent"].clip(upper=5.0)
    audit_df["feature_name"] = audit_df["category_id"].apply(offered_category_to_feature)

    wide_df = (
        audit_df.pivot_table(
            index=id_cols,
            columns="feature_name",
            values="offered_percent",
            aggfunc="max",
            fill_value=0.0,
        )
        .reset_index()
    )
    wide_df.columns.name = None

    # Добавляем отсутствующие must-have offered-фичи нулями, чтобы схема была стабильной.
    for col in offered_feature_columns:
        if col not in wide_df.columns:
            wide_df[col] = 0.0

    return wide_df, audit_df


def apply_offers_to_inference_frame(
    df,
    offers_wide,
    offered_feature_columns,
    client_col="cdi_id",
    month_col="forecast_month",
    drop_existing=True,
):
    """
    Подмешивает generated offers в inference-датафрейм и пересчитывает агрегаты.

    Если drop_existing=True, старые offered_percent_cat_* удаляются и заменяются
    сгенерированными. Это нужно, чтобы модель видела именно новые предложения.
    """
    result = df.copy()
    offered_feature_columns = get_offered_feature_columns(offered_feature_columns)

    merge_cols = [client_col]
    if month_col is not None and month_col in result.columns and month_col in offers_wide.columns:
        merge_cols.append(month_col)

    if drop_existing:
        result = result.drop(columns=[col for col in offered_feature_columns if col in result.columns], errors="ignore")

    result = result.merge(offers_wide, on=merge_cols, how="left")

    for col in offered_feature_columns:
        if col not in result.columns:
            result[col] = 0.0
    result[offered_feature_columns] = result[offered_feature_columns].fillna(0.0).astype("float32")

    result["num_offered_categories"] = (result[offered_feature_columns] > 0).sum(axis=1).astype("int16")
    result["max_offered_percent"] = result[offered_feature_columns].max(axis=1).astype("float32")
    result["avg_offered_percent"] = (
        result[offered_feature_columns]
        .replace(0, np.nan)
        .mean(axis=1)
        .fillna(0.0)
        .astype("float32")
    )

    return result


def save_offers_audit(audit_df, path):
    """Сохраняет audit-таблицу назначенных категорий."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(path, index=False)
    return path
