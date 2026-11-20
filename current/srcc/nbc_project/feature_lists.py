"""Загрузка списков фичей, вынесенных из старых ноутбуков."""

from pathlib import Path
import json


def _default_config_path():
    return Path(__file__).resolve().parents[2] / "configs" / "feature_lists.json"


def load_feature_lists(config_path=None):
    """Читает все списки фичей из json."""
    path = Path(config_path) if config_path is not None else _default_config_path()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_feature_list(name, config_path=None):
    """Возвращает конкретный список фичей по имени."""
    data = load_feature_lists(config_path)
    if name not in data:
        raise KeyError(f"Список фичей {name!r} не найден. Доступно: {sorted(data)}")
    return data[name]


def select_existing_columns(df, feature_list_name, extra_cols=None, config_path=None):
    """
    Берёт из df только те фичи из списка, которые реально есть в датафрейме.
    extra_cols нужны для id, периода и таргета.
    """
    extra_cols = list(extra_cols or [])
    features = get_feature_list(feature_list_name, config_path=config_path)
    columns = []
    for col in extra_cols + features:
        if col in df.columns and col not in columns:
            columns.append(col)
    return df[columns].copy()
