"""Взвешенный гибрид на пуле кандидатов с весами по типам пользователей.

Скор пары (u, i) пула: score = Σ_m w_m(type_u) · s̃_m(u, i), где s̃_m — скор
компоненты m, нормированный min-max внутри пула пользователя, type_u — new (не
более θ оценок в train) или active. Компоненты — скоры моделей (Popularity,
Content-Based, Item-based CF, PureSVD, FunkSVD) и affinity пользователя к
категории книги. Top-K выбирается по скору пула, при заданных квотах — с
переранжированием (`recsys.balancing`); при нехватке кандидатов список
дополняется книгами Popularity.
"""

import copy
import time
from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
from scipy import sparse

from recsys import balancing, candidates, config, features
from recsys.balancing import Quotas
from recsys.candidates import CandidatePool
from recsys.data import interaction_matrix
from recsys.models.base import Recommender

USER_TYPES: tuple[str, ...] = ("new", "active")
AFFINITY: str = "affinity"
FALLBACK_MODEL: str = "popularity"


@dataclass(frozen=True)
class ScoredPool:
    """Пул кандидатов со скорами компонент.

    Attributes:
        pool: Пул кандидатов.
        values: Исходные скоры компонент формы (n_pairs, n_components).
        normalized: Скоры, нормированные min-max внутри пула пользователя.
        user_types: Код типа пользователя (индекс в `USER_TYPES`) формы (n_users,).
        components: Названия компонент в порядке столбцов.
        seconds: Время формирования пула и скоров компонент.
    """

    pool: CandidatePool
    values: np.ndarray
    normalized: np.ndarray
    user_types: np.ndarray
    components: tuple[str, ...]
    seconds: float = 0.0


def normalize_per_user(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Нормирует скоры min-max внутри пула каждого пользователя по каждому столбцу.

    Нечисловые скоры (`-inf`, NaN) и столбцы с постоянным скором в пуле
    пользователя получают 0.

    Args:
        values: Скоры формы (n_pairs, n_components), пары упорядочены по пользователю.
        offsets: Границы пар пользователей формы (n_users + 1,).

    Returns:
        Нормированные скоры float32 в [0, 1].
    """
    sizes = np.diff(offsets)
    nonempty = sizes > 0
    if not nonempty.any():
        return np.zeros(values.shape, dtype=np.float32)
    finite = np.isfinite(values)
    starts = offsets[:-1][nonempty]
    low = np.minimum.reduceat(np.where(finite, values, np.inf), starts, axis=0)
    high = np.maximum.reduceat(np.where(finite, values, -np.inf), starts, axis=0)
    low = np.repeat(low, sizes[nonempty], axis=0)
    span = np.repeat(high, sizes[nonempty], axis=0) - low
    valid = finite & np.isfinite(span) & (span > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = (values - low) / span
    return np.where(valid, normalized, 0.0).astype(np.float32)


def weight_matrix(weights: dict[str, dict[str, float]], components: tuple[str, ...]) -> np.ndarray:
    """Преобразует веса по типам пользователей в матрицу.

    Args:
        weights: Веса компонент по типу пользователя; отсутствующая компонента — 0.
        components: Названия компонент в порядке столбцов.

    Returns:
        Матрица (len(USER_TYPES), n_components) float32.

    Raises:
        ValueError: При неизвестной компоненте или типе пользователя.
    """
    matrix = np.zeros((len(USER_TYPES), len(components)), dtype=np.float32)
    for user_type, type_weights in weights.items():
        unknown = set(type_weights) - set(components)
        if user_type not in USER_TYPES or unknown:
            raise ValueError(f"неизвестный тип пользователя или компоненты: {user_type}, {unknown}")
        for name, weight in type_weights.items():
            matrix[USER_TYPES.index(user_type), components.index(name)] = weight
    return matrix


def combine(scored: ScoredPool, weights: dict[str, dict[str, float]]) -> np.ndarray:
    """Считает взвешенный скор пар пула.

    Args:
        scored: Пул со скорами компонент.
        weights: Веса компонент по типу пользователя.

    Returns:
        Скор каждой пары формы (n_pairs,).
    """
    matrix = weight_matrix(weights, scored.components)
    pair_weights = matrix[scored.user_types[scored.pool.rows]]
    return np.einsum("ij,ij->i", scored.normalized, pair_weights)


def rank_pool(pool: CandidatePool, scores: np.ndarray) -> np.ndarray:
    """Упорядочивает кандидатов каждого пользователя по убыванию скора.

    Args:
        pool: Пул кандидатов.
        scores: Скор каждой пары формы (n_pairs,).

    Returns:
        `book_id` формы (n_users, max размер пула); -1 — пустая позиция; при
        равенстве скоров — меньший `book_id`.
    """
    rows = pool.rows
    order = np.lexsort((-scores, rows))
    positions = np.arange(order.size) - pool.offsets[rows[order]]
    width = int(pool.sizes.max(initial=0))
    ranked = np.full((pool.user_ids.size, width), -1, dtype=np.int64)
    ranked[rows[order], positions] = pool.book_ids[order]
    return ranked


def fill_fallback(
    recommendations: np.ndarray,
    user_ids: np.ndarray,
    fallback_order: np.ndarray,
    seen: sparse.csr_matrix,
) -> np.ndarray:
    """Дополняет неполные списки книгами по порядку fallback-модели.

    Пустые позиции находятся в конце списка; в дополнение не попадают
    прочитанные книги и книги, уже стоящие в списке.

    Args:
        recommendations: `book_id` формы (n_users, K), -1 — пустая позиция.
        user_ids: Идентификаторы пользователей строк.
        fallback_order: Индексы книг (`book_id - 1`) по убыванию скора fallback.
        seen: Матрица (N_USERS, N_BOOKS) прочитанных книг.

    Returns:
        Списки с заполненными позициями (если хватает книг fallback).
    """
    result = recommendations.copy()
    short = np.flatnonzero((result < 0).any(axis=1))
    if short.size == 0:
        return result
    k = result.shape[1]
    short_seen = seen[np.asarray(user_ids)[short] - 1]
    width = min(2 * k + int(short_seen.getnnz(axis=1).max()), fallback_order.size)
    books = np.broadcast_to(fallback_order[:width], (short.size, width))
    row_keys = np.arange(short.size, dtype=np.int64)[:, None] * config.N_BOOKS
    seen_rows, seen_cols = short_seen.nonzero()
    current = result[short]
    blocked_keys = np.concatenate(
        [seen_rows * config.N_BOOKS + seen_cols, (row_keys + current - 1)[current > 0]]
    )
    allowed = ~np.isin(row_keys + books, blocked_keys)
    rank = np.cumsum(allowed, axis=1)
    filled = (current > 0).sum(axis=1)
    take = allowed & (rank <= (k - filled)[:, None])
    rows, cols = np.nonzero(take)
    current[rows, filled[rows] + rank[rows, cols] - 1] = books[rows, cols] + 1
    result[short] = current
    return result


class HybridRecommender(Recommender):
    """Взвешенный гибрид моделей на пуле кандидатов.

    Attributes:
        models: Модели-компоненты по названию; должна быть модель `popularity`
            (fallback).
        categories: Категории книг (`features.book_categories`).
        sources: Названия моделей — источников кандидатов.
        weights: Веса компонент по типу пользователя (`new`, `active`).
        n_candidates: Число кандидатов от каждого источника.
        quotas: Квоты балансировки; None — top-K по скору.
        new_max_ratings: θ — максимум оценок в train у нового пользователя.
        components: Названия компонент: модели и `affinity`.
    """

    def __init__(
        self,
        models: dict[str, Recommender],
        categories: pd.Series,
        sources: tuple[str, ...] = config.CANDIDATE_SOURCES,
        weights: dict[str, dict[str, float]] = config.HYBRID_WEIGHTS,
        n_candidates: int = config.CANDIDATES_PER_SOURCE,
        quotas: Quotas | None = None,
        new_max_ratings: int = config.USER_NEW_MAX_RATINGS,
        name: str = "Hybrid",
    ) -> None:
        """Задаёт компоненты и параметры гибрида.

        Args:
            models: Модели-компоненты по названию.
            categories: Категории книг.
            sources: Названия моделей — источников кандидатов.
            weights: Веса компонент по типу пользователя.
            n_candidates: Число кандидатов от каждого источника.
            quotas: Квоты балансировки; None — без балансировки.
            new_max_ratings: θ — порог нового пользователя.
            name: Отображаемое название.

        Raises:
            ValueError: Если нет модели `popularity` или источник не входит в модели.
        """
        if FALLBACK_MODEL not in models or not set(sources) <= set(models):
            raise ValueError("нужна модель popularity, источники должны входить в модели")
        self.models = models
        self.categories = categories
        self.sources = tuple(sources)
        self.components = (*models, AFFINITY)
        weight_matrix(weights, self.components)
        self.weights = weights
        self.n_candidates = n_candidates
        self.quotas = quotas
        self.new_max_ratings = new_max_ratings
        self.name = name
        self._category_codes = categories.cat.codes.to_numpy().astype(np.int64)

    def with_params(self, **params: object) -> Self:
        """Возвращает копию гибрида с изменёнными параметрами и общим обученным состоянием.

        Args:
            **params: Новые значения атрибутов (`weights`, `quotas`, `name`).

        Returns:
            Копия гибрида.

        Raises:
            ValueError: При неизвестном параметре.
        """
        unknown = set(params) - {"weights", "quotas", "name"}
        if unknown:
            raise ValueError(f"неизвестные параметры: {unknown}")
        if "weights" in params:
            weight_matrix(params["weights"], self.components)
        hybrid = copy.copy(self)
        for key, value in params.items():
            setattr(hybrid, key, value)
        return hybrid

    def fit(self, train: pd.DataFrame, fit_models: bool = True) -> Self:
        """Обучает компоненты и считает статистики train.

        Args:
            train: Обучающие оценки `user_id, book_id, rating`.
            fit_models: Обучать модели; False — модели уже обучены на `train`.

        Returns:
            Обученный гибрид.
        """
        if fit_models:
            for model in self.models.values():
                model.fit(train)
        self.seen = interaction_matrix(train, binary=True)
        self.user_n_ratings = np.bincount(train["user_id"].to_numpy() - 1, minlength=config.N_USERS)
        self.affinity = features.category_affinity(train, self.categories)
        item_counts = np.bincount(train["book_id"].to_numpy() - 1, minlength=config.N_BOOKS)
        self.is_head = balancing.head_mask(item_counts)
        popularity = np.asarray(self.models[FALLBACK_MODEL].score(np.array([1])))[0]
        order = np.argsort(-np.where(np.isfinite(popularity), popularity, -np.inf), kind="stable")
        self.fallback_order = order[np.isfinite(popularity[order])]
        return self

    def user_types(self, user_ids: np.ndarray) -> np.ndarray:
        """Определяет тип пользователей по числу оценок в train.

        Args:
            user_ids: Идентификаторы пользователей.

        Returns:
            Код типа (индекс в `USER_TYPES`): 0 — new, 1 — active.
        """
        n_ratings = self.user_n_ratings[np.asarray(user_ids) - 1]
        return np.where(n_ratings <= self.new_max_ratings, 0, 1)

    def _exclusion(self, exclude: sparse.csr_matrix | None) -> sparse.csr_matrix:
        """Объединяет прочитанное в train с дополнительно исключаемыми книгами."""
        return self.seen if exclude is None else self.seen.maximum(exclude.astype(bool)).tocsr()

    def prepare(
        self,
        user_ids: np.ndarray,
        exclude: sparse.csr_matrix | None = None,
        batch_size: int = config.EVAL_BATCH_SIZE,
    ) -> ScoredPool:
        """Формирует пул кандидатов и скоры компонент для пользователей.

        Матрицы скоров моделей батча считаются один раз и используются и для
        выбора кандидатов, и для скоров компонент.

        Args:
            user_ids: Идентификаторы пользователей.
            exclude: Дополнительно исключаемые книги (N_USERS, N_BOOKS); книги
                из train исключаются всегда.
            batch_size: Число пользователей в батче.

        Returns:
            Пул со скорами компонент.
        """
        start_time = time.perf_counter()
        user_ids = np.asarray(user_ids)
        seen = self._exclusion(exclude)
        parts = []
        for start in range(0, user_ids.size, batch_size):
            batch = user_ids[start : start + batch_size]
            scores = {name: model.score(batch) for name, model in self.models.items()}
            rows, items, bits = candidates.select_candidates(
                {name: scores[name] for name in self.sources}, seen[batch - 1], self.n_candidates
            )
            columns = [np.asarray(scores[name])[rows, items] for name in self.models]
            columns.append(self.affinity[batch[rows] - 1, self._category_codes[items]])
            parts.append((rows + start, items, bits, np.column_stack(columns)))
        rows, items, bits, values = (np.concatenate(arrays) for arrays in zip(*parts, strict=True))
        pool = candidates.build_pool(user_ids, rows, items, bits, self.sources)
        return ScoredPool(
            pool=pool,
            values=values.astype(np.float32),
            normalized=normalize_per_user(values, pool.offsets),
            user_types=self.user_types(user_ids),
            components=self.components,
            seconds=time.perf_counter() - start_time,
        )

    def rank(
        self, scored: ScoredPool, k: int, exclude: sparse.csr_matrix | None = None
    ) -> np.ndarray:
        """Формирует top-K по готовому пулу с текущими весами и квотами.

        Квоты соблюдаются для списка длины K, поэтому при балансировке список
        строится отдельно для каждой оцениваемой длины.

        Args:
            scored: Пул со скорами компонент (`prepare`).
            k: Длина списка.
            exclude: Те же дополнительно исключаемые книги, что в `prepare`.

        Returns:
            `book_id` формы (n_users, K); -1 — пустая позиция.
        """
        ranked = rank_pool(scored.pool, combine(scored, self.weights))
        if self.quotas is None:
            top = np.full((ranked.shape[0], k), -1, dtype=np.int64)
            top[:, : min(k, ranked.shape[1])] = ranked[:, :k]
        else:
            top = balancing.rerank(ranked, k, self._category_codes, self.is_head, self.quotas)
        return fill_fallback(
            top, scored.pool.user_ids, self.fallback_order, self._exclusion(exclude)
        )

    def recommend(
        self,
        user_ids: np.ndarray,
        k: int,
        exclude: sparse.csr_matrix | None = None,
        batch_size: int = config.EVAL_BATCH_SIZE,
    ) -> np.ndarray:
        """Формирует top-K рекомендации гибрида.

        Args:
            user_ids: Идентификаторы пользователей.
            k: Длина списка.
            exclude: Дополнительно исключаемые книги; книги из train исключаются всегда.
            batch_size: Число пользователей в батче.

        Returns:
            Массив `book_id` формы (len(user_ids), K); -1 — пустая позиция.
        """
        user_ids = np.asarray(user_ids)
        batches = [
            self.rank(self.prepare(user_ids[start : start + batch_size], exclude), k, exclude)
            for start in range(0, user_ids.size, batch_size)
        ]
        return np.vstack(batches) if batches else np.empty((0, k), dtype=np.int64)

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        """Считает гибридный скор кандидатов пула (без квот и fallback).

        Args:
            user_ids: Идентификаторы пользователей.

        Returns:
            Матрица (len(user_ids), N_BOOKS) float32; `-inf` — книга вне пула.
        """
        scored = self.prepare(user_ids)
        result = np.full((len(user_ids), config.N_BOOKS), -np.inf, dtype=np.float32)
        result[scored.pool.rows, scored.pool.book_ids - 1] = combine(scored, self.weights)
        return result

    def explain(self, user_id: int, books: pd.DataFrame, k: int = 10) -> pd.DataFrame:
        """Показывает top-K пользователя с источниками, категориями и скорами компонент.

        Args:
            user_id: Идентификатор пользователя.
            books: Метаданные книг (`book_id`, `title`).
            k: Длина списка.

        Returns:
            DataFrame по позициям: `book_id`, `title`, `category`, `head`,
            `sources` (пусто — книга из fallback), нормированные скоры компонент
            и итоговый скор `score`.
        """
        scored = self.prepare(np.array([user_id]))
        top = self.rank(scored, k)[0]
        top = top[top > 0]
        pair = pd.Series(np.arange(scored.pool.book_ids.size), index=scored.pool.book_ids)
        pairs = pair.reindex(top).to_numpy()
        in_pool = ~np.isnan(pairs)
        index = np.where(in_pool, pairs, 0).astype(np.int64)
        labels = scored.pool.source_labels(index)
        table = pd.DataFrame(
            {
                "book_id": top,
                "title": books.set_index("book_id").loc[top, "title"].to_numpy(),
                "category": self.categories.loc[top].to_numpy(),
                "head": self.is_head[top - 1],
                "sources": [label if ok else "" for label, ok in zip(labels, in_pool, strict=True)],
            },
            index=pd.RangeIndex(1, top.size + 1, name="position"),
        )
        components = pd.DataFrame(
            np.where(in_pool[:, None], scored.normalized[index], np.nan),
            columns=list(scored.components),
            index=table.index,
        )
        score = np.where(in_pool, combine(scored, self.weights)[index], np.nan)
        return pd.concat([table, components], axis=1).assign(score=score)
