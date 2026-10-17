"""Проверка per-user temporal split: состав частей, порядок, отсутствие утечки."""

import numpy as np
import pandas as pd
import pytest

from recsys.split import temporal_split


@pytest.fixture
def ratings() -> pd.DataFrame:
    """Перемежающиеся оценки: пользователь 1 — 10, 2 — 3, 3 — 7 оценок."""
    users = [1, 3, 1, 2, 1, 3, 1, 1, 3, 2, 1, 3, 1, 3, 1, 2, 3, 1, 3, 1]
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "user_id": users,
            "book_id": np.arange(1, len(users) + 1),
            "rating": rng.integers(1, 6, len(users)),
        }
    )


def test_parts_are_disjoint_and_complete(ratings: pd.DataFrame) -> None:
    split = temporal_split(ratings)
    assert split.train.index.intersection(split.test.index).empty
    assert split.train.index.union(split.test.index).equals(ratings.index)


def test_test_size_per_user(ratings: pd.DataFrame) -> None:
    split = temporal_split(ratings, test_fraction=0.2, min_ratings=5)
    sizes = split.test["user_id"].value_counts().to_dict()
    # 10 · 0.2 = 2; 7 · 0.2 = 1.4 → 1; у пользователя 2 (3 оценки) test пуст.
    assert sizes == {1: 2, 3: 1}


def test_test_rows_are_latest(ratings: pd.DataFrame) -> None:
    split = temporal_split(ratings)
    last_train = split.train.reset_index().groupby("user_id")["index"].max()
    first_test = split.test.reset_index().groupby("user_id")["index"].min()
    assert (first_test > last_train.loc[first_test.index]).all()


def test_every_test_user_has_train(ratings: pd.DataFrame) -> None:
    split = temporal_split(ratings)
    assert set(split.test["user_id"]) <= set(split.train["user_id"])


def test_nested_split_uses_train_only(ratings: pd.DataFrame) -> None:
    outer = temporal_split(ratings)
    inner = temporal_split(outer.train)
    assert inner.test.index.isin(outer.train.index).all()
    assert not inner.test.index.isin(outer.test.index).any()


def test_invalid_fraction(ratings: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        temporal_split(ratings, test_fraction=1.0)
