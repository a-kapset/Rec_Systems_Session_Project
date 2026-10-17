"""Проверка пула кандидатов: исключение прочитанного, объединение источников, полнота."""

from typing import Self

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from recsys import candidates, config, evaluation
from recsys.models.base import Recommender


class FixedScores(Recommender):
    """Модель с заданными скорами: строка `user_id - 1` матрицы `scores`."""

    def __init__(self, scores: np.ndarray, name: str = "fixed") -> None:
        """Запоминает матрицу скоров (n_users, N_BOOKS)."""
        self.scores = scores
        self.name = name

    def fit(self, train: pd.DataFrame) -> Self:
        """Не требует обучения."""
        return self

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        """Возвращает строки заданной матрицы."""
        return self.scores[np.asarray(user_ids) - 1].copy()


def _scores(rows: dict[int, dict[int, float]], n_users: int = 2) -> np.ndarray:
    """Строит матрицу скоров: нули, кроме заданных значений (строка → книга → скор)."""
    matrix = np.zeros((n_users, config.N_BOOKS), dtype=np.float32)
    for row, values in rows.items():
        for item, value in values.items():
            matrix[row, item] = value
    return matrix


@pytest.fixture
def sources() -> dict[str, FixedScores]:
    """Источник a: у пользователя 1 скоры книг 0–4, у пользователя 2 — постоянный ноль;
    источник b: у пользователя 1 скоры книг 3–5."""
    return {
        "a": FixedScores(_scores({0: {0: 5, 1: 4, 2: 3, 3: 2, 4: 1}})),
        "b": FixedScores(_scores({0: {3: 3, 4: 2, 5: 1}})),
    }


def _seen() -> sparse.csr_matrix:
    """Пользователь 1 прочитал книгу с индексом 0."""
    return sparse.csr_matrix(([1.0], ([0], [0])), shape=(config.N_USERS, config.N_BOOKS))


def test_select_candidates_excludes_seen_and_merges_sources(
    sources: dict[str, FixedScores],
) -> None:
    scores = {name: model.score(np.array([1, 2])) for name, model in sources.items()}
    rows, items, bits = candidates.select_candidates(scores, _seen()[:2], n=3)
    # a: книги 1, 2, 3 (книга 0 прочитана); b: книги 3, 4, 5; у пользователя 2
    # скоры постоянны — кандидатов нет.
    np.testing.assert_array_equal(rows, [0, 0, 0, 0, 0])
    np.testing.assert_array_equal(items, [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(bits, [1, 1, 3, 2, 2])


def test_candidates_strictly_above_user_minimum() -> None:
    scores = {"a": _scores({0: {7: 2.0}}, n_users=1)}
    seen = sparse.csr_matrix((1, config.N_BOOKS))
    _, items, _ = candidates.select_candidates(scores, seen, n=5)
    np.testing.assert_array_equal(items, [7])


def test_generate_pool(sources: dict[str, FixedScores]) -> None:
    pool = candidates.generate(sources, np.array([1, 2]), _seen(), n=3, batch_size=1)
    np.testing.assert_array_equal(pool.offsets, [0, 5, 5])
    np.testing.assert_array_equal(pool.book_ids, [2, 3, 4, 5, 6])
    np.testing.assert_array_equal(pool.from_source("b"), [False, False, True, True, True])
    assert pool.source_labels(np.array([0, 2])) == ["a", "a+b"]
    assert pool.composition().to_dict() == {"a": 0.4, "b": 0.4, "a+b": 0.2}
    assert not _seen()[pool.user_ids[pool.rows] - 1, pool.book_ids - 1].any()


def test_pool_metrics(sources: dict[str, FixedScores]) -> None:
    train = pd.DataFrame({"user_id": [1, 2], "book_id": [1, 1], "rating": [5, 5]})
    holdout = pd.DataFrame({"user_id": [1, 1, 2], "book_id": [4, 9, 2], "rating": [5, 4, 4]})
    eval_set = evaluation.build_eval_set(train, holdout, n_users=None)
    pool = candidates.generate(sources, eval_set.user_ids, eval_set.seen, n=3)
    table = candidates.pool_metrics(pool, eval_set)
    # Пользователь 1: релевантны книги 4 (источники a и b) и 9 (нет в пуле) → 1/2;
    # пользователь 2: пул пуст → 0.
    assert table.loc["pool", "recall"] == pytest.approx(0.25)
    assert table.loc["a", "recall"] == pytest.approx(0.25)
    assert table.loc["pool", "candidates"] == pytest.approx(2.5)
    assert table.loc["pool", "unique_share"] == pytest.approx(0.0)
