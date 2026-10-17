"""Неперсонализированная модель Popularity.

Ранжирование книг по статистикам train: средний рейтинг среди книг с числом
оценок не меньше порога, weighted rating (Bayesian average) со сглаживанием
к глобальному среднему, число оценок. Список одинаков для всех пользователей,
из него исключаются книги, уже оценённые пользователем.
"""

from typing import Literal, Self

import numpy as np
import pandas as pd

from recsys import config
from recsys.models.base import Recommender

Method = Literal["mean", "weighted", "count"]


class Popularity(Recommender):
    """Top-N самых популярных книг.

    Варианты скора книги i с числом оценок v_i и средним R_i:
    `mean` — R_i при v_i ≥ min_ratings, иначе книга не рекомендуется;
    `weighted` — WR_i = v_i / (v_i + m) · R_i + m / (v_i + m) · C, где C —
    глобальное среднее, m = min_ratings;
    `count` — v_i.

    Attributes:
        method: Вариант скора.
        min_ratings: Порог числа оценок (`mean`) или сила сглаживания m (`weighted`).
        stats: Статистики книг после `fit`: `n_ratings`, `mean_rating`, `score`.
    """

    def __init__(self, method: Method = "mean", min_ratings: int = 0) -> None:
        """Задаёт вариант модели.

        Args:
            method: Вариант скора: `mean`, `weighted` или `count`.
            min_ratings: Порог числа оценок (`mean`) или сила сглаживания (`weighted`).
        """
        self.method = method
        self.min_ratings = min_ratings
        suffix = "" if method == "count" else f", m={min_ratings}"
        self.name = f"Popularity[{method}{suffix}]"
        self.stats: pd.DataFrame | None = None
        self._scores: np.ndarray | None = None

    def fit(self, train: pd.DataFrame) -> Self:
        """Считает статистики книг и скор популярности.

        Args:
            train: Обучающие оценки `user_id, book_id, rating`.

        Returns:
            Обученная модель.

        Raises:
            ValueError: При неизвестном варианте скора.
        """
        book_index = train["book_id"].to_numpy() - 1
        n_ratings = np.bincount(book_index, minlength=config.N_BOOKS).astype(np.float64)
        rating_sum = np.bincount(book_index, weights=train["rating"], minlength=config.N_BOOKS)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_rating = rating_sum / n_ratings
        if self.method == "mean":
            scores = np.where(n_ratings >= max(self.min_ratings, 1), mean_rating, -np.inf)
        elif self.method == "weighted":
            global_mean = rating_sum.sum() / n_ratings.sum()
            m = self.min_ratings
            scores = (rating_sum + m * global_mean) / (n_ratings + m)
            scores = np.where(n_ratings > 0, scores, -np.inf)
        elif self.method == "count":
            scores = np.where(n_ratings > 0, n_ratings, -np.inf)
        else:
            raise ValueError(f"неизвестный вариант Popularity: {self.method}")
        self._scores = scores.astype(np.float32)
        self.stats = pd.DataFrame(
            {"n_ratings": n_ratings.astype(np.int64), "mean_rating": mean_rating, "score": scores},
            index=pd.RangeIndex(1, config.N_BOOKS + 1, name="book_id"),
        )
        return self

    @property
    def n_candidates(self) -> int:
        """Число книг, допущенных к рекомендации."""
        return int(np.isfinite(self._scores).sum())

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        """Возвращает одинаковые скоры книг для всех пользователей батча.

        Args:
            user_ids: Идентификаторы пользователей.

        Returns:
            Матрица (len(user_ids), N_BOOKS).
        """
        return np.broadcast_to(self._scores, (len(user_ids), config.N_BOOKS))

    def top_n(self, books: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Возвращает неперсонализированный top-N с метаданными книг.

        Args:
            books: Метаданные книг (`book_id`, `title`, `authors`).
            n: Длина списка.

        Returns:
            DataFrame: `book_id`, `title`, `authors`, `n_ratings`, `mean_rating`, `score`.
        """
        top = self.stats[np.isfinite(self.stats["score"])].nlargest(n, "score")
        return top.reset_index().merge(
            books[["book_id", "title", "authors"]], on="book_id", how="left"
        )[["book_id", "title", "authors", "n_ratings", "mean_rating", "score"]]
