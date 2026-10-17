"""Матричные разложения: FunkSVD (`surprise.SVD`) и PureSVD (усечённый SVD).

FunkSVD — r̂(u, i) = μ + b_u + b_i + q_iᵀp_u, обучение SGD только по наблюдаемым
оценкам; параметры извлекаются из `surprise` в массивы, индексированные
`user_id - 1` и `book_id - 1`, предсказание и скоринг выполняются матричными
операциями без поэлементных вызовов `predict`. PureSVD — усечённое
разложение R ≈ U_k Σ_k V_kᵀ разреженной матрицы оценок с нулями на месте
пропусков (`scipy.sparse.linalg.svds`), скор — проекция r_u V_k V_kᵀ.
"""

from typing import Self

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds
from surprise import SVD, BaselineOnly, Dataset, Reader, Trainset

from recsys import config
from recsys.data import interaction_matrix
from recsys.models.base import Recommender


def build_trainset(train: pd.DataFrame) -> Trainset:
    """Преобразует таблицу оценок в `surprise.Trainset`.

    Args:
        train: Оценки `user_id, book_id, rating`.

    Returns:
        Обучающая выборка `surprise` по всем оценкам.
    """
    reader = Reader(rating_scale=(config.RATING_MIN, config.RATING_MAX))
    data = Dataset.load_from_df(train[["user_id", "book_id", "rating"]], reader)
    return data.build_full_trainset()


def _raw_indices(trainset: Trainset) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает индексы `user_id - 1` и `book_id - 1` в порядке inner id `surprise`.

    Args:
        trainset: Обучающая выборка `surprise`.

    Returns:
        Индексы пользователей и книг.
    """
    users = np.array([trainset.to_raw_uid(i) for i in trainset.all_users()]) - 1
    items = np.array([trainset.to_raw_iid(i) for i in trainset.all_items()]) - 1
    return users, items


def baseline_predict(train: pd.DataFrame, user_ids: np.ndarray, book_ids: np.ndarray) -> np.ndarray:
    """Обучает `surprise.BaselineOnly` (μ + b_u + b_i, ALS) и предсказывает оценки.

    Args:
        train: Обучающие оценки `user_id, book_id, rating`.
        user_ids: Идентификаторы пользователей.
        book_ids: Идентификаторы книг той же длины.

    Returns:
        Оценки float32, ограниченные шкалой [1, 5].
    """
    trainset = build_trainset(train)
    user_bias_inner, item_bias_inner = BaselineOnly(verbose=False).fit(trainset).compute_baselines()
    users, items = _raw_indices(trainset)
    user_bias = np.zeros(config.N_USERS)
    item_bias = np.zeros(config.N_BOOKS)
    user_bias[users] = user_bias_inner
    item_bias[items] = item_bias_inner
    estimate = (
        trainset.global_mean
        + user_bias[np.asarray(user_ids) - 1]
        + item_bias[np.asarray(book_ids) - 1]
    )
    return np.clip(estimate, config.RATING_MIN, config.RATING_MAX).astype(np.float32)


class FunkSVD(Recommender):
    """FunkSVD (`surprise.SVD`) с векторизованными предсказанием и top-N.

    Для пользователя или книги, отсутствующих в train, латентный вектор и
    смещение равны нулю — как в `surprise`: для неизвестного пользователя
    r̂ = μ + b_i. Top-N — книги с наибольшей предсказанной оценкой среди книг,
    имеющих в train не менее `min_item_ratings` оценок.

    Attributes:
        n_factors: Размерность латентных векторов.
        n_epochs: Число эпох SGD.
        lr_all: Шаг SGD для всех параметров.
        reg_all: Коэффициент L2-регуляризации для всех параметров.
        random_state: Seed инициализации латентных векторов.
        min_item_ratings: Минимальное число оценок книги в train для top-N.
        global_mean: μ после `fit`.
        user_bias: b_u формы (N_USERS,).
        item_bias: b_i формы (N_BOOKS,).
        user_factors: p_u формы (N_USERS, n_factors).
        item_factors: q_i формы (N_BOOKS, n_factors).
    """

    def __init__(
        self,
        n_factors: int = config.SVD_N_FACTORS,
        n_epochs: int = config.SVD_N_EPOCHS,
        lr_all: float = config.SVD_LR_ALL,
        reg_all: float = config.SVD_REG_ALL,
        random_state: int = config.RANDOM_STATE,
        min_item_ratings: int = config.SVD_MIN_ITEM_RATINGS,
    ) -> None:
        """Задаёт гиперпараметры модели.

        Args:
            n_factors: Размерность латентных векторов.
            n_epochs: Число эпох SGD.
            lr_all: Шаг SGD.
            reg_all: Коэффициент L2-регуляризации.
            random_state: Seed инициализации.
            min_item_ratings: Минимальное число оценок книги в train для top-N.
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state
        self.min_item_ratings = min_item_ratings

    @property
    def name(self) -> str:
        """Название модели с параметрами."""
        return (
            f"SVD[f={self.n_factors}, epochs={self.n_epochs}, "
            f"lr={self.lr_all:g}, reg={self.reg_all:g}, min_ratings={self.min_item_ratings}]"
        )

    def fit(self, train: pd.DataFrame) -> Self:
        """Обучает `surprise.SVD` и извлекает параметры в массивы по ID.

        Args:
            train: Обучающие оценки `user_id, book_id, rating`.

        Returns:
            Обученная модель.
        """
        trainset = build_trainset(train)
        algo = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=self.random_state,
        ).fit(trainset)

        users, items = _raw_indices(trainset)
        self.global_mean = float(trainset.global_mean)
        self.user_bias = np.zeros(config.N_USERS, dtype=np.float32)
        self.item_bias = np.zeros(config.N_BOOKS, dtype=np.float32)
        self.user_factors = np.zeros((config.N_USERS, self.n_factors), dtype=np.float32)
        self.item_factors = np.zeros((config.N_BOOKS, self.n_factors), dtype=np.float32)
        self.user_bias[users] = algo.bu
        self.item_bias[items] = algo.bi
        self.user_factors[users] = algo.pu
        self.item_factors[items] = algo.qi
        self.item_counts = np.bincount(train["book_id"].to_numpy() - 1, minlength=config.N_BOOKS)
        self.algo = algo
        return self.set_min_item_ratings(self.min_item_ratings)

    def set_min_item_ratings(self, min_item_ratings: int) -> Self:
        """Меняет порог числа оценок книги для top-N без переобучения.

        Args:
            min_item_ratings: Новый порог; книги без оценок исключаются всегда.

        Returns:
            Модель с обновлённым набором кандидатов.
        """
        self.min_item_ratings = min_item_ratings
        self.candidates = self.item_counts >= max(min_item_ratings, 1)
        return self

    def predict(self, user_ids: np.ndarray, book_ids: np.ndarray) -> np.ndarray:
        """Предсказывает оценки для пар (пользователь, книга).

        Args:
            user_ids: Идентификаторы пользователей.
            book_ids: Идентификаторы книг той же длины.

        Returns:
            Оценки float32, ограниченные шкалой [1, 5].
        """
        users = np.asarray(user_ids) - 1
        items = np.asarray(book_ids) - 1
        estimate = (
            self.global_mean
            + self.user_bias[users]
            + self.item_bias[items]
            + np.einsum("ij,ij->i", self.user_factors[users], self.item_factors[items])
        )
        return np.clip(estimate, config.RATING_MIN, config.RATING_MAX).astype(np.float32)

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        """Считает предсказанные оценки всех книг для батча пользователей.

        Скоры не ограничиваются шкалой, чтобы не создавать равенств на
        границе 5.

        Args:
            user_ids: Идентификаторы пользователей.

        Returns:
            Матрица (len(user_ids), N_BOOKS) float32; `-inf` — книга не входит
            в кандидаты.
        """
        users = np.asarray(user_ids) - 1
        scores = self.user_factors[users] @ self.item_factors.T
        scores += self.global_mean + self.user_bias[users, None] + self.item_bias[None, :]
        scores[:, ~self.candidates] = -np.inf
        return scores

    def get_recommendations(
        self,
        user_id: int,
        books: pd.DataFrame,
        n: int = 5,
        exclude: sparse.csr_matrix | None = None,
    ) -> pd.DataFrame:
        """Формирует top-N книг с наибольшей предсказанной оценкой.

        Args:
            user_id: Идентификатор пользователя.
            books: Метаданные книг (`book_id`, `title`, `authors`).
            n: Длина списка.
            exclude: Матрица (N_USERS, N_BOOKS) книг, исключаемых из выдачи.

        Returns:
            DataFrame: `book_id`, `title`, `authors`, `predicted_rating`.
        """
        result = super().get_recommendations(user_id, books, n, exclude)
        return result.rename(columns={"score": "predicted_rating"})


class PureSVD(Recommender):
    """PureSVD: усечённый SVD матрицы оценок для top-N.

    R ≈ U_k Σ_k V_kᵀ, пропуски — нули; скор книг для пользователя —
    r_u V_k V_kᵀ (проекция строки оценок на k-мерное пространство книг).
    Оценки 1–5 не предсказываются: модель восстанавливает матрицу, где
    неоценённые книги равны 0.

    Attributes:
        n_factors: Ранг разложения k.
        singular_values: Сингулярные числа по убыванию после `fit`.
        item_factors: V_k формы (N_BOOKS, k).
    """

    def __init__(
        self, n_factors: int = config.PURE_SVD_N_FACTORS, random_state: int = config.RANDOM_STATE
    ) -> None:
        """Задаёт ранг разложения.

        Args:
            n_factors: Ранг k.
            random_state: Seed начального вектора `svds`.
        """
        self.n_factors = n_factors
        self.random_state = random_state

    @property
    def name(self) -> str:
        """Название модели с параметрами."""
        return f"PureSVD[k={self.n_factors}]"

    def fit(self, train: pd.DataFrame) -> Self:
        """Выполняет усечённое SVD разреженной матрицы оценок.

        Args:
            train: Обучающие оценки `user_id, book_id, rating`.

        Returns:
            Обученная модель.
        """
        self._ratings = interaction_matrix(train)
        _, singular_values, vt = svds(
            self._ratings, k=self.n_factors, random_state=self.random_state
        )
        order = np.argsort(-singular_values)
        self.singular_values = singular_values[order]
        self.item_factors = np.ascontiguousarray(vt[order].T, dtype=np.float32)
        self.known_items = self._ratings.getnnz(axis=0) > 0
        return self

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        """Считает скоры r_u V_k V_kᵀ для батча пользователей.

        Args:
            user_ids: Идентификаторы пользователей.

        Returns:
            Матрица (len(user_ids), N_BOOKS) float32; `-inf` — книга без оценок в train.
        """
        profiles = self._ratings[np.asarray(user_ids) - 1] @ self.item_factors
        scores = profiles @ self.item_factors.T
        scores[:, ~self.known_items] = -np.inf
        return scores
