"""Per-user temporal split оценок на обучающую и отложенную части.

Порядок строк `ratings.csv` соответствует хронологии, поэтому для каждого
пользователя в отложенную часть попадает доля его последних по порядку оценок.
Повторное применение к train даёт разбиение train_inner / valid для подбора
гиперпараметров без обращения к test.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recsys import config


@dataclass(frozen=True)
class Split:
    """Результат разбиения оценок.

    Attributes:
        train: Обучающая часть в исходном порядке строк.
        test: Отложенная часть в исходном порядке строк.
    """

    train: pd.DataFrame
    test: pd.DataFrame


def temporal_split(
    ratings: pd.DataFrame,
    test_fraction: float = config.TEST_FRACTION,
    min_ratings: int = config.MIN_RATINGS_TO_SPLIT,
) -> Split:
    """Разбивает оценки каждого пользователя по порядку строк.

    Число отложенных оценок пользователя — `round(n · test_fraction)`, но не
    менее 1. Пользователи с числом оценок меньше `min_ratings` целиком остаются
    в train.

    Args:
        ratings: Оценки `user_id, book_id, rating` в хронологическом порядке.
        test_fraction: Доля последних оценок пользователя в отложенной части.
        min_ratings: Минимальное число оценок пользователя для разбиения.

    Returns:
        Разбиение `Split` с сохранением исходных индексов строк.

    Raises:
        ValueError: Если `test_fraction` вне интервала (0, 1).
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction должен лежать в интервале (0, 1)")
    groups = ratings.groupby("user_id", sort=False)
    position = groups.cumcount().to_numpy()
    n_user = groups["user_id"].transform("size").to_numpy()
    n_test = np.maximum(1, np.floor(n_user * test_fraction + 0.5)).astype(np.int64)
    is_test = (position >= n_user - n_test) & (n_user >= min_ratings)
    return Split(train=ratings[~is_test], test=ratings[is_test])
