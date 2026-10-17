"""Проверка оценки по готовым спискам: пользовательские метрики, сегменты, exposure."""

import numpy as np
import pandas as pd
import pytest

from recsys import config, evaluation


@pytest.fixture
def eval_set() -> evaluation.EvalSet:
    """Три пользователя; релевантные книги: {1, 2}, {3}, {4, 5, 6}."""
    train = pd.DataFrame({"user_id": [1, 2, 3], "book_id": [7, 7, 8], "rating": [5, 4, 3]})
    holdout = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3, 3, 3],
            "book_id": [1, 2, 3, 4, 5, 6],
            "rating": [5, 4, 4, 5, 4, 4],
        }
    )
    return evaluation.build_eval_set(train, holdout, n_users=None)


@pytest.fixture
def recommendations() -> np.ndarray:
    """Top-2: пользователь 1 — 1 попадание на позиции 1, 2 — 0, 3 — 2 попадания."""
    return np.array([[1, 9], [9, 10], [4, 5]])


def test_user_metrics_match_ranking_metrics(
    eval_set: evaluation.EvalSet, recommendations: np.ndarray
) -> None:
    per_user = evaluation.user_metrics(recommendations, eval_set, k=2)
    total = evaluation.ranking_metrics(recommendations, eval_set, ks=(2,)).loc[2]
    np.testing.assert_allclose(per_user["precision"], [0.5, 0.0, 1.0])
    np.testing.assert_allclose(per_user["hit_rate"], [1.0, 0.0, 1.0])
    for column in ("precision", "recall", "ndcg", "map", "hit_rate", "novelty"):
        assert per_user[column].mean() == pytest.approx(total[column])


def test_align_rows_and_subset(eval_set: evaluation.EvalSet, recommendations: np.ndarray) -> None:
    target = evaluation.subset(eval_set, np.array([3, 1]))
    np.testing.assert_array_equal(target.user_ids, [1, 3])
    aligned = evaluation.align_rows(recommendations, eval_set, target)
    np.testing.assert_array_equal(aligned, [[1, 9], [4, 5]])
    np.testing.assert_array_equal(target.n_relevant, [2, 3])
    other = evaluation.EvalSet(np.array([4]), target.relevant[:1], target.seen, target.item_counts)
    with pytest.raises(ValueError):
        evaluation.align_rows(recommendations, eval_set, other)


def test_segment_metrics(eval_set: evaluation.EvalSet, recommendations: np.ndarray) -> None:
    segments = pd.Series(["a", "b", "a"], index=[1, 2, 3], dtype="category")
    table = evaluation.segment_metrics({"m": recommendations}, eval_set, segments, k=2)
    assert table.loc[("m", "a"), "n_users"] == 2
    assert table.loc[("m", "a"), "hit_rate"] == pytest.approx(1.0)
    assert table.loc[("m", "b"), "hit_rate"] == pytest.approx(0.0)
    assert table.loc[("m", "a"), "n_relevant"] == pytest.approx(2.5)
    assert table.index.get_level_values("segment")[:2].tolist() == ["a", "b"]


def test_exposure_shares() -> None:
    labels = np.where(np.arange(1, config.N_BOOKS + 1) <= 2, "head", "tail")
    item_segments = pd.Series(
        pd.Categorical(labels, categories=["head", "mid", "tail"]),
        index=np.arange(1, config.N_BOOKS + 1),
    )
    recs = np.array([[1, 2, 3], [1, 5, -1]])
    table = evaluation.exposure({"m": recs}, item_segments, k=3)
    assert table.loc["m", "head"] == pytest.approx(3 / 5)
    assert table.loc["m", "mid"] == 0.0
    assert table.loc["m", "tail"] == pytest.approx(2 / 5)
