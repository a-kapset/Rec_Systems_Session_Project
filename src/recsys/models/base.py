"""Общий интерфейс top-N моделей рекомендаций.

Модель обучается на таблице оценок и возвращает плотную матрицу скоров для
батча пользователей; базовый класс превращает скоры в top-K списки с
исключением книг, уже оценённых пользователем в train.
"""

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import pandas as pd
from scipy import sparse

from recsys import config


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Выбирает индексы K наибольших скоров в каждой строке по убыванию.

    Позиции с нечисловым скором (`-inf`, NaN) заменяются на -1.

    Args:
        scores: Скоры формы (n_users, n_items).
        k: Длина списка.

    Returns:
        Индексы объектов формы (n_users, min(K, n_items)).
    """
    scores = np.where(np.isnan(scores), -np.inf, scores)
    k = min(k, scores.shape[1])
    candidates = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    candidate_scores = np.take_along_axis(scores, candidates, axis=1)
    order = np.argsort(-candidate_scores, axis=1, kind="stable")
    top = np.take_along_axis(candidates, order, axis=1)
    top_scores = np.take_along_axis(candidate_scores, order, axis=1)
    return np.where(np.isfinite(top_scores), top, -1)


class Recommender(ABC):
    """Базовый класс top-N модели.

    Индекс книги в матрицах — `book_id - 1`, индекс пользователя — `user_id - 1`.
    """

    name: str = "base"

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> Self:
        """Обучает модель.

        Args:
            train: Обучающие оценки `user_id, book_id, rating`.

        Returns:
            Обученная модель.
        """

    @abstractmethod
    def score(self, user_ids: np.ndarray) -> np.ndarray:
        """Считает скоры всех книг для батча пользователей.

        Args:
            user_ids: Идентификаторы пользователей.

        Returns:
            Матрица (len(user_ids), N_BOOKS) float32; больше — лучше,
            `-inf` или NaN — книга не рекомендуется.
        """

    def recommend(
        self,
        user_ids: np.ndarray,
        k: int,
        exclude: sparse.csr_matrix | None = None,
        batch_size: int = config.EVAL_BATCH_SIZE,
    ) -> np.ndarray:
        """Формирует top-K рекомендации для пользователей.

        Args:
            user_ids: Идентификаторы пользователей.
            k: Длина списка.
            exclude: Матрица (N_USERS, N_BOOKS) книг, исключаемых из выдачи
                (обычно — оценки из train).
            batch_size: Число пользователей в батче.

        Returns:
            Массив `book_id` формы (len(user_ids), K); -1 — пустая позиция.
        """
        user_ids = np.asarray(user_ids)
        result = np.full((user_ids.size, k), -1, dtype=np.int64)
        for start in range(0, user_ids.size, batch_size):
            batch = user_ids[start : start + batch_size]
            scores = np.array(self.score(batch), dtype=np.float32)
            if exclude is not None:
                rows, cols = exclude[batch - 1].nonzero()
                scores[rows, cols] = -np.inf
            top = top_k_indices(scores, k)
            result[start : start + batch.size, : top.shape[1]] = np.where(top >= 0, top + 1, -1)
        return result

    def get_recommendations(
        self,
        user_id: int,
        books: pd.DataFrame,
        n: int = 5,
        exclude: sparse.csr_matrix | None = None,
    ) -> pd.DataFrame:
        """Формирует top-N рекомендации одного пользователя с метаданными книг.

        Args:
            user_id: Идентификатор пользователя.
            books: Метаданные книг (`book_id`, `title`, `authors`).
            n: Длина списка.
            exclude: Матрица (N_USERS, N_BOOKS) книг, исключаемых из выдачи.

        Returns:
            DataFrame: `book_id`, `title`, `authors`, `score` по убыванию скора.
        """
        scores = np.array(self.score(np.array([user_id])), dtype=np.float32)
        if exclude is not None:
            scores[0, exclude[user_id - 1].indices] = -np.inf
        top = top_k_indices(scores, n)[0]
        top = top[top >= 0]
        result = books.set_index("book_id").loc[top + 1, ["title", "authors"]].reset_index()
        return result.assign(score=scores[0, top])
