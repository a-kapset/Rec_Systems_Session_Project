"""Item-based Collaborative Filtering на разреженной матрице оценок.

Схожесть книг — cosine по векторам оценок (исходным, центрированным по среднему
пользователя — adjusted cosine — или по среднему книги — Pearson), считается
разреженным произведением `XᵀX` блоками строк, умножается на shrinkage
n_ij / (n_ij + λ) и хранится в `csr_matrix`. Оценка (u, i) предсказывается
по K книгам, наиболее похожим на i, среди оценённых пользователем u; top-N
строится разреженным произведением профиля пользователя на матрицу top-K
соседей каждой книги.
"""

from typing import Literal, Self

import numpy as np
import pandas as pd
from scipy import sparse

from recsys import config
from recsys.data import interaction_matrix
from recsys.models.base import Recommender, top_k_indices

Similarity = Literal["cosine", "adjusted_cosine", "pearson"]
Ranking = Literal["rating", "similarity"]


def _center(matrix: sparse.csr_matrix, axis: int) -> sparse.csr_matrix:
    """Вычитает из ненулевых элементов среднее по строке или столбцу.

    Args:
        matrix: Матрица оценок (N_USERS, N_BOOKS).
        axis: 1 — среднее пользователя (строки), 0 — среднее книги (столбцы).

    Returns:
        Матрица той же структуры с центрированными значениями.
    """
    counts = matrix.getnnz(axis=axis)
    means = np.asarray(matrix.sum(axis=axis)).ravel() / np.maximum(counts, 1)
    centered = matrix.copy()
    offsets = np.repeat(means, counts) if axis == 1 else means[matrix.indices]
    centered.data = (centered.data - offsets).astype(np.float32)
    return centered


def item_similarity(
    ratings: sparse.csr_matrix,
    kind: Similarity = "adjusted_cosine",
    shrinkage: float = 0.0,
    positive_only: bool = True,
    batch_size: int = 1_000,
) -> sparse.csr_matrix:
    """Считает матрицу схожести книг по оценкам пользователей.

    s(i, j) = x_iᵀx_j / (‖x_i‖·‖x_j‖) · n_ij / (n_ij + λ), где x_i — столбец
    оценок книги (после центрирования), n_ij — число пользователей, оценивших
    обе книги. Плотными создаются только блоки `batch_size × N_BOOKS`.

    Args:
        ratings: Матрица оценок (N_USERS, N_BOOKS) из train.
        kind: `cosine` — исходные оценки, `adjusted_cosine` — центрирование по
            пользователю, `pearson` — центрирование по книге.
        shrinkage: λ; 0 — без shrinkage.
        positive_only: Отбрасывать отрицательные схожести.
        batch_size: Число строк матрицы схожести в блоке.

    Returns:
        Матрица (N_BOOKS, N_BOOKS) float32, диагональ пустая.
    """
    if kind == "adjusted_cosine":
        values = _center(ratings, axis=1)
    elif kind == "pearson":
        values = _center(ratings, axis=0)
    else:
        values = ratings.astype(np.float32)
    values_t = values.T.tocsr()
    norms = np.sqrt(np.asarray(values_t.multiply(values_t).sum(axis=1)).ravel())
    inv_norms = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    binary = (ratings != 0).astype(np.float32)
    binary_t = binary.T.tocsr()
    n_items = ratings.shape[1]
    blocks = []
    for start in range(0, n_items, batch_size):
        stop = min(start + batch_size, n_items)
        block = (values_t[start:stop] @ values).toarray()
        block *= inv_norms[start:stop, None] * inv_norms[None, :]
        if shrinkage > 0:
            co_counts = (binary_t[start:stop] @ binary).toarray()
            block *= co_counts / (co_counts + shrinkage)
        rows = np.arange(stop - start)
        block[rows, rows + start] = 0.0
        if positive_only:
            block[block < 0] = 0.0
        blocks.append(sparse.csr_matrix(block, dtype=np.float32))
    return sparse.vstack(blocks, format="csr")


def prune_top_k(similarity: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    """Оставляет в каждой строке K наибольших значений.

    Args:
        similarity: Матрица схожести (строка — целевая книга, столбец — сосед).
        k: Число соседей.

    Returns:
        Матрица той же формы, не более K ненулевых значений в строке.
    """
    coo = similarity.tocoo()
    order = np.lexsort((-coo.data, coo.row))
    row_sorted = coo.row[order]
    rank = np.arange(order.size) - similarity.indptr[row_sorted]
    keep = order[rank < k]
    return sparse.csr_matrix(
        (coo.data[keep], (coo.row[keep], coo.col[keep])), shape=similarity.shape, dtype=np.float32
    )


def sparse_nbytes(matrix: sparse.csr_matrix) -> int:
    """Возвращает объём памяти CSR-матрицы в байтах.

    Args:
        matrix: Матрица `csr_matrix`.

    Returns:
        Суммарный размер массивов `data`, `indices`, `indptr`.
    """
    return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes


class ItemCF(Recommender):
    """Item-based Collaborative Filtering.

    Предсказание оценки с центрированием по пользователю:
    r̂(u, i) = r̄_u + Σ_j s(i, j)·(r_uj − r̄_u) / Σ_j |s(i, j)|, j — K книг,
    наиболее похожих на i (по |s| при учёте отрицательных схожестей), среди
    оценённых u; без центрирования — Σ_j s(i, j)·r_uj / Σ_j |s(i, j)|. При
    неотрицательных схожестях оба варианта совпадают алгебраически. При
    отсутствии соседей — r̄_u, затем r̄_i, затем глобальное среднее μ; результат
    ограничивается шкалой [1, 5].

    Attributes:
        similarity_kind: Мера схожести.
        n_neighbors: K — число соседей.
        shrinkage: λ — сила shrinkage.
        centered: Центрирование оценок по пользователю в предсказании.
        use_negative: Учёт отрицательных схожестей в предсказании оценки.
        ranking: Скор top-N: `rating` — предсказанная оценка по соседям из
            top-K схожих с i, `similarity` — сумма схожестей с книгами,
            оценёнными пользователем на 4–5.
        min_support: Минимальное число соседей среди оценённых пользователем
            для ранжирования по `rating`.
        similarity: Матрица схожести (N_BOOKS, N_BOOKS) после `fit`.
        neighbors: Матрица top-K соседей каждой книги после `fit`.
    """

    def __init__(
        self,
        similarity: Similarity = config.ITEM_CF_SIMILARITY,
        n_neighbors: int = config.ITEM_CF_NEIGHBORS,
        shrinkage: float = config.ITEM_CF_SHRINKAGE,
        centered: bool = True,
        use_negative: bool = False,
        ranking: Ranking = "similarity",
        min_support: int = 1,
    ) -> None:
        """Задаёт параметры модели.

        Args:
            similarity: Мера схожести: `cosine`, `adjusted_cosine`, `pearson`.
            n_neighbors: K — число соседей.
            shrinkage: λ — сила shrinkage.
            centered: Центрирование по пользователю в предсказании.
            use_negative: Учёт отрицательных схожестей в предсказании оценки.
            ranking: Скор top-N: `rating` или `similarity`.
            min_support: Минимальное число соседей для ранжирования по `rating`.
        """
        self.similarity_kind = similarity
        self.n_neighbors = n_neighbors
        self.shrinkage = shrinkage
        self.centered = centered
        self.use_negative = use_negative
        self.ranking = ranking
        self.min_support = min_support
        self.similarity: sparse.csr_matrix | None = None
        self.neighbors: sparse.csr_matrix | None = None

    @property
    def name(self) -> str:
        """Название модели с параметрами."""
        suffix = f", support={self.min_support}" if self.ranking == "rating" else ""
        return (
            f"Item-CF[{self.similarity_kind}, K={self.n_neighbors}, λ={self.shrinkage:g}, "
            f"{self.ranking}{suffix}]"
        )

    def fit(self, train: pd.DataFrame) -> Self:
        """Считает средние, матрицу схожести и top-K соседей книг.

        Args:
            train: Обучающие оценки `user_id, book_id, rating`.

        Returns:
            Обученная модель.
        """
        ratings = interaction_matrix(train)
        user_counts = np.diff(ratings.indptr)
        item_counts = ratings.getnnz(axis=0)
        self.global_mean = float(ratings.data.mean())
        self.user_means = np.where(
            user_counts > 0,
            np.asarray(ratings.sum(axis=1)).ravel() / np.maximum(user_counts, 1),
            np.nan,
        ).astype(np.float32)
        self.item_means = np.where(
            item_counts > 0,
            np.asarray(ratings.sum(axis=0)).ravel() / np.maximum(item_counts, 1),
            np.nan,
        ).astype(np.float32)
        self._ratings = ratings
        self._liked = interaction_matrix(
            train[train["rating"] >= config.RELEVANCE_THRESHOLD], binary=True
        )
        self._deviations = _center(ratings, axis=1) if self.centered else ratings

        # Оценённые книги пользователя в виде дополненных массивов для predict.
        width = int(user_counts.max())
        positions = np.arange(ratings.nnz) - np.repeat(ratings.indptr[:-1], user_counts)
        rows = np.repeat(np.arange(ratings.shape[0]), user_counts)
        self._rated_items = np.full((ratings.shape[0], width), -1, dtype=np.int32)
        self._rated_values = np.zeros((ratings.shape[0], width), dtype=np.float32)
        self._rated_items[rows, positions] = ratings.indices
        self._rated_values[rows, positions] = self._deviations.data

        self.similarity = item_similarity(
            ratings, self.similarity_kind, self.shrinkage, positive_only=not self.use_negative
        )
        return self.set_n_neighbors(self.n_neighbors)

    def set_n_neighbors(self, n_neighbors: int) -> Self:
        """Меняет число соседей K без пересчёта матрицы схожести.

        Args:
            n_neighbors: Новое значение K.

        Returns:
            Модель с обновлённой матрицей top-K соседей.
        """
        self.n_neighbors = n_neighbors
        positive = self.similarity.multiply(self.similarity > 0).tocsr()
        self.neighbors = prune_top_k(positive, n_neighbors)
        self._neighbors_t = self.neighbors.T.tocsr()
        return self

    def _fallback(self, user_index: np.ndarray, item_index: np.ndarray) -> np.ndarray:
        """Возвращает оценку без соседей: r̄_u, затем r̄_i, затем μ.

        Args:
            user_index: Индексы пользователей.
            item_index: Индексы книг.

        Returns:
            Массив оценок float32.
        """
        fallback = self.user_means[user_index]
        fallback = np.where(np.isnan(fallback), self.item_means[item_index], fallback)
        return np.where(np.isnan(fallback), self.global_mean, fallback).astype(np.float32)

    def predict(
        self,
        user_ids: np.ndarray,
        book_ids: np.ndarray,
        item_batch: int = 500,
        pair_batch: int = 100_000,
    ) -> np.ndarray:
        """Предсказывает оценки для пар (пользователь, книга).

        Пары группируются по книге; для блока книг строки матрицы схожести
        выгружаются в плотный вид, схожести с оценёнными книгами пользователя
        выбираются индексированием, K соседей — `argpartition`.

        Args:
            user_ids: Идентификаторы пользователей.
            book_ids: Идентификаторы книг той же длины.
            item_batch: Число целевых книг в блоке.
            pair_batch: Максимальное число пар в батче.

        Returns:
            Предсказанные оценки float32 в исходном порядке пар.
        """
        user_index = np.asarray(user_ids) - 1
        item_index = np.asarray(book_ids) - 1
        prediction = self._fallback(user_index, item_index)
        order = np.argsort(item_index, kind="stable")
        sorted_items = item_index[order]
        k = self.n_neighbors
        for item_start in range(0, config.N_BOOKS, item_batch):
            lo, hi = np.searchsorted(sorted_items, [item_start, item_start + item_batch])
            if lo == hi:
                continue
            block = self.similarity[item_start : item_start + item_batch].toarray()
            for start in range(lo, hi, pair_batch):
                pairs = order[start : min(start + pair_batch, hi)]
                users = user_index[pairs]
                rated = self._rated_items[users]
                sims = block[(item_index[pairs] - item_start)[:, None], rated]
                sims[rated < 0] = 0.0
                values = self._rated_values[users]
                if sims.shape[1] > k:
                    top = np.argpartition(-np.abs(sims), k - 1, axis=1)[:, :k]
                    sims = np.take_along_axis(sims, top, axis=1)
                    values = np.take_along_axis(values, top, axis=1)
                weight = np.abs(sims).sum(axis=1)
                has_neighbors = weight > 0
                estimate = (sims * values).sum(axis=1) / np.where(has_neighbors, weight, 1.0)
                if self.centered:
                    estimate += self.user_means[users]
                prediction[pairs] = np.where(has_neighbors, estimate, prediction[pairs])
        return np.clip(prediction, config.RATING_MIN, config.RATING_MAX)

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        """Считает скоры всех книг для батча пользователей.

        `similarity`: Σ s(i, j) по книгам j из top-K соседей i, оценённым на 4–5.
        `rating`: предсказанная оценка по соседям j из top-K схожих с i,
        оценённым пользователем; книги с числом таких соседей меньше
        `min_support` не рекомендуются.

        Args:
            user_ids: Идентификаторы пользователей.

        Returns:
            Матрица (len(user_ids), N_BOOKS).
        """
        users = np.asarray(user_ids) - 1
        neighbors_t = self._neighbors_t
        if self.ranking == "similarity":
            return (self._liked[users] @ neighbors_t).toarray()
        rated = (self._ratings[users] != 0).astype(np.float32)
        weight = (rated @ neighbors_t).toarray()
        support = (rated @ (neighbors_t != 0).astype(np.float32)).toarray()
        numerator = (self._deviations[users] @ neighbors_t).toarray()
        with np.errstate(invalid="ignore", divide="ignore"):
            estimate = numerator / weight
        if self.centered:
            estimate += self.user_means[users, None]
        return np.where(support >= max(self.min_support, 1), estimate, -np.inf).astype(np.float32)

    def get_similar_books(self, book_id: int, books: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Находит N книг с наибольшей схожестью по оценкам.

        Args:
            book_id: Идентификатор книги.
            books: Метаданные книг (`book_id`, `title`, `authors`).
            n: Число похожих книг.

        Returns:
            DataFrame: `book_id`, `title`, `authors`, `similarity`.
        """
        row = self.similarity[book_id - 1].toarray()
        row[row <= 0] = -np.inf
        top = top_k_indices(row, n)[0]
        top = top[top >= 0]
        result = books.set_index("book_id").loc[top + 1, ["title", "authors"]].reset_index()
        return result.assign(similarity=row[0, top])
