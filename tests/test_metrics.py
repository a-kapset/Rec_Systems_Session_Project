"""Проверка метрик на игрушечных примерах с ручным расчётом."""

import numpy as np
import pytest
from scipy import sparse

from recsys import metrics


def _relevant(rows: list[list[int]], n_items: int = 10) -> sparse.csr_matrix:
    """Строит бинарную матрицу релевантности по спискам объектов.

    Args:
        rows: Релевантные объекты каждого пользователя.
        n_items: Размер каталога.

    Returns:
        Матрица (len(rows), n_items).
    """
    matrix = np.zeros((len(rows), n_items), dtype=np.float32)
    for user, items in enumerate(rows):
        matrix[user, items] = 1.0
    return sparse.csr_matrix(matrix)


@pytest.fixture
def example() -> tuple[np.ndarray, np.ndarray]:
    """Два пользователя: R,N,R,N,R при |rel| = 3 и N,R,–,–,– при |rel| = 2."""
    recs = np.array([[0, 1, 2, 3, 4], [5, 6, -1, -1, -1]])
    relevant = _relevant([[0, 2, 4], [6, 9]])
    hits = metrics.hit_matrix(recs, relevant)
    return hits, np.diff(relevant.indptr)


def test_hit_matrix(example: tuple[np.ndarray, np.ndarray]) -> None:
    hits, n_relevant = example
    expected = [[True, False, True, False, True], [False, True, False, False, False]]
    np.testing.assert_array_equal(hits, expected)
    np.testing.assert_array_equal(n_relevant, [3, 2])


def test_padding_does_not_match_previous_user() -> None:
    # Ключ пустой позиции второго пользователя совпал бы с объектом 9 первого.
    hits = metrics.hit_matrix(np.array([[9], [-1]]), _relevant([[9], [0]]))
    np.testing.assert_array_equal(hits, [[True], [False]])


def test_precision_recall(example: tuple[np.ndarray, np.ndarray]) -> None:
    hits, n_relevant = example
    np.testing.assert_allclose(metrics.precision_at_k(hits, 5), [3 / 5, 1 / 5])
    np.testing.assert_allclose(metrics.recall_at_k(hits, n_relevant, 5), [1.0, 1 / 2])
    # При |rel| > K знаменатель Recall — все релевантные: 1 / 3, а не 1 / 2.
    np.testing.assert_allclose(metrics.recall_at_k(hits, n_relevant, 2), [1 / 3, 1 / 2])
    np.testing.assert_allclose(metrics.precision_at_k(hits, 2), [1 / 2, 1 / 2])


def test_ndcg(example: tuple[np.ndarray, np.ndarray]) -> None:
    hits, n_relevant = example
    dcg_0 = 1 + 1 / np.log2(4) + 1 / np.log2(6)
    idcg_0 = 1 + 1 / np.log2(3) + 1 / np.log2(4)
    ndcg_1 = (1 / np.log2(3)) / (1 + 1 / np.log2(3))
    np.testing.assert_allclose(metrics.ndcg_at_k(hits, n_relevant, 5), [dcg_0 / idcg_0, ndcg_1])


def test_ndcg_depends_on_order() -> None:
    relevant = _relevant([[0], [0]])
    hits = metrics.hit_matrix(np.array([[0, 1], [1, 0]]), relevant)
    n_relevant = np.diff(relevant.indptr)
    np.testing.assert_allclose(metrics.ndcg_at_k(hits, n_relevant, 2), [1.0, 1 / np.log2(3)])
    np.testing.assert_allclose(metrics.precision_at_k(hits, 2), [0.5, 0.5])


def test_idcg_uses_all_relevant() -> None:
    # Одно попадание на первой позиции при трёх релевантных: nDCG@2 < 1.
    relevant = _relevant([[0, 5, 6]])
    hits = metrics.hit_matrix(np.array([[0, 1]]), relevant)
    ndcg = metrics.ndcg_at_k(hits, np.diff(relevant.indptr), 2)
    np.testing.assert_allclose(ndcg, [1 / (1 + 1 / np.log2(3))])


def test_average_precision_and_hit_rate(example: tuple[np.ndarray, np.ndarray]) -> None:
    hits, n_relevant = example
    ap = metrics.average_precision_at_k(hits, n_relevant, 5)
    np.testing.assert_allclose(ap, [(1 + 2 / 3 + 3 / 5) / 3, (1 / 2) / 2])
    # Знаменатель AP@K — min(K, |rel|).
    np.testing.assert_allclose(metrics.average_precision_at_k(hits, n_relevant, 1), [1.0, 0.0])
    np.testing.assert_allclose(metrics.hit_rate_at_k(hits, 1), [1.0, 0.0])
    np.testing.assert_allclose(metrics.hit_rate_at_k(hits, 5), [1.0, 1.0])


def test_requires_relevant_items() -> None:
    hits = np.zeros((1, 3), dtype=bool)
    with pytest.raises(ValueError):
        metrics.ndcg_at_k(hits, np.array([0]), 3)


def test_coverage_novelty_popularity() -> None:
    recs = np.array([[0, 1], [1, -1]])
    assert metrics.coverage_at_k(recs, n_catalog=4, k=2) == pytest.approx(0.5)
    counts = np.array([3, 1])
    information = -np.log2(np.array([4, 2]) / 6)
    np.testing.assert_allclose(
        metrics.novelty_at_k(recs, counts, 2), [information.mean(), information[1]]
    )
    np.testing.assert_allclose(metrics.mean_popularity_at_k(recs, counts, 2), [2.0, 1.0])


def test_rmse_mae() -> None:
    y_true, y_pred = np.array([4, 2, 5]), np.array([3, 2, 3])
    assert metrics.rmse(y_true, y_pred) == pytest.approx(np.sqrt(5 / 3))
    assert metrics.mae(y_true, y_pred) == pytest.approx(1.0)
