"""Проверка выбора top-K, исключения просмотренного и моделей Popularity."""

import numpy as np
import pandas as pd
import pytest

from recsys.data import interaction_matrix
from recsys.models.base import top_k_indices
from recsys.models.popularity import Popularity


def test_top_k_indices_order_and_padding() -> None:
    scores = np.array([[0.1, 0.9, -np.inf, 0.5], [np.nan, -np.inf, 2.0, -np.inf]])
    np.testing.assert_array_equal(top_k_indices(scores, 3), [[1, 3, 0], [2, -1, -1]])


def _train() -> pd.DataFrame:
    """Книга 1: 3 оценки со средним 4; книга 2: одна оценка 5; книга 3: 2 оценки по 2."""
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 1, 2, 3],
            "book_id": [1, 1, 1, 2, 3, 3],
            "rating": [4, 4, 4, 5, 2, 2],
        }
    )


def test_popularity_variants() -> None:
    train = _train()
    mean_all = Popularity("mean", min_ratings=0).fit(train)
    assert mean_all.top_n(pd.DataFrame(columns=["book_id", "title", "authors"]), 3)[
        "book_id"
    ].tolist() == [2, 1, 3]
    mean_threshold = Popularity("mean", min_ratings=2).fit(train)
    assert np.isneginf(mean_threshold.stats.loc[2, "score"])
    weighted = Popularity("weighted", min_ratings=2).fit(train)
    global_mean = 21 / 6
    assert weighted.stats.loc[2, "score"] == pytest.approx((5 + 2 * global_mean) / 3)
    count = Popularity("count").fit(train)
    assert count.stats.loc[1, "score"] == 3


def test_recommend_excludes_seen_items() -> None:
    train = _train()
    model = Popularity("count").fit(train)
    recs = model.recommend(np.array([1, 3]), k=3, exclude=interaction_matrix(train))
    # Пользователь 1 оценил книги 1 и 2, пользователь 3 — книги 1 и 3.
    np.testing.assert_array_equal(recs, [[3, -1, -1], [2, -1, -1]])
