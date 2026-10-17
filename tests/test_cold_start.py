"""Проверка симуляции cold start: усечение истории, скрытие книг, метрики новых книг."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from recsys import cold_start, config, evaluation


def test_truncate_history_and_hide_items() -> None:
    train = pd.DataFrame(
        {"user_id": [1, 2, 1, 1, 2], "book_id": [1, 2, 3, 4, 5], "rating": [5, 4, 3, 2, 1]}
    )
    truncated = cold_start.truncate_history(train, np.array([1]), n=2)
    assert truncated["book_id"].tolist() == [1, 2, 3, 5]
    ratings = pd.DataFrame({"user_id": 1, "book_id": np.arange(1, config.N_BOOKS + 1), "rating": 4})
    visible, hidden = cold_start.hide_items(ratings, share=0.01)
    assert hidden.size == 100
    assert not visible["book_id"].isin(hidden).any()
    assert len(visible) == config.N_BOOKS - 100


def test_new_item_metrics() -> None:
    relevant = sparse.csr_matrix(
        (np.ones(4), ([0, 0, 1, 2], [0, 4, 5, 1])), shape=(3, config.N_BOOKS)
    )
    eval_set = evaluation.EvalSet(
        np.array([1, 2, 3]), relevant, relevant, np.ones(config.N_BOOKS, dtype=np.int64)
    )
    recs = np.array([[5, 1], [7, 9], [2, -1]])
    result = cold_start.new_item_metrics(recs, eval_set, np.array([5, 6, 7]), k=2)
    assert result["users_with_new"] == 2
    assert result["exposure"] == pytest.approx(2 / 5)
    assert result["recall_new"] == pytest.approx(0.5)
    assert result["hit_rate_new"] == pytest.approx(0.5)


def test_truncate_history_per_user_and_mixed_lengths() -> None:
    train = pd.DataFrame(
        {"user_id": [1, 2, 1, 2, 1, 3], "book_id": [1, 2, 3, 4, 5, 6], "rating": 4}
    )
    truncated = cold_start.truncate_history(train, np.array([1, 2]), np.array([2, 1]))
    assert truncated["book_id"].tolist() == [1, 2, 3, 6]
    lengths = cold_start.mixed_lengths(10, (1, 3))
    assert sorted(lengths.tolist()) == [1] * 5 + [3] * 5
