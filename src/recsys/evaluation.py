"""Оценка top-N моделей на отложенной выборке.

Единый для всех моделей набор оценки: фиксированная случайная выборка
пользователей с хотя бы одной релевантной книгой (оценка ≥ порога) в отложенной
части, матрица релевантных книг, матрица train для исключения уже оценённых книг
и популярность книг в train. Рекомендации формируются батчами, метрики
считаются векторизованно и усредняются по пользователям (macro).
"""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from recsys import config, metrics
from recsys.data import interaction_matrix
from recsys.models.base import Recommender


@dataclass(frozen=True)
class EvalSet:
    """Данные для оценки top-N моделей.

    Attributes:
        user_ids: Идентификаторы оцениваемых пользователей.
        relevant: Бинарная матрица (len(user_ids), N_BOOKS) релевантных книг.
        seen: Матрица (N_USERS, N_BOOKS) оценок train, исключаемых из выдачи.
        item_counts: Число оценок каждой книги в train.
    """

    user_ids: np.ndarray
    relevant: sparse.csr_matrix
    seen: sparse.csr_matrix
    item_counts: np.ndarray

    @property
    def n_relevant(self) -> np.ndarray:
        """Число релевантных книг каждого оцениваемого пользователя."""
        return np.diff(self.relevant.indptr)

    @property
    def n_catalog(self) -> int:
        """Число книг, имеющих оценки в train."""
        return int((self.item_counts > 0).sum())


def build_eval_set(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    n_users: int | None = config.EVAL_N_USERS,
    threshold: int = config.RELEVANCE_THRESHOLD,
    random_state: int = config.RANDOM_STATE,
) -> EvalSet:
    """Формирует набор оценки по разбиению train / holdout.

    Args:
        train: Обучающие оценки.
        holdout: Отложенные оценки (valid или test).
        n_users: Размер случайной выборки пользователей; None — все пользователи.
        threshold: Порог релевантности оценки.
        random_state: Seed выборки пользователей.

    Returns:
        Набор оценки `EvalSet`.
    """
    relevant_all = interaction_matrix(holdout[holdout["rating"] >= threshold], binary=True)
    candidates = np.flatnonzero(np.diff(relevant_all.indptr) > 0) + 1
    if n_users is not None and n_users < candidates.size:
        rng = np.random.default_rng(random_state)
        candidates = np.sort(rng.choice(candidates, size=n_users, replace=False))
    item_counts = np.bincount(train["book_id"].to_numpy() - 1, minlength=config.N_BOOKS)
    return EvalSet(
        user_ids=candidates,
        relevant=relevant_all[candidates - 1],
        seen=interaction_matrix(train, binary=True),
        item_counts=item_counts,
    )


def ranking_metrics(
    recommendations: np.ndarray, eval_set: EvalSet, ks: tuple[int, ...] = config.K_VALUES
) -> pd.DataFrame:
    """Считает метрики top-N по готовым спискам рекомендаций.

    Args:
        recommendations: `book_id` формы (n_users, ≥ max(ks)), -1 — пустая позиция;
            строки выровнены с `eval_set.user_ids`.
        eval_set: Набор оценки.
        ks: Длины списков.

    Returns:
        DataFrame с индексом K и колонками метрик; `short_lists` — доля
        списков короче K.
    """
    items = np.where(recommendations > 0, recommendations - 1, -1)
    hits = metrics.hit_matrix(items, eval_set.relevant)
    n_relevant = eval_set.n_relevant
    rows = {}
    for k in ks:
        rows[k] = {
            "precision": metrics.precision_at_k(hits, k).mean(),
            "recall": metrics.recall_at_k(hits, n_relevant, k).mean(),
            "ndcg": metrics.ndcg_at_k(hits, n_relevant, k).mean(),
            "map": metrics.average_precision_at_k(hits, n_relevant, k).mean(),
            "hit_rate": metrics.hit_rate_at_k(hits, k).mean(),
            "coverage": metrics.coverage_at_k(items, eval_set.n_catalog, k),
            "novelty": metrics.novelty_at_k(items, eval_set.item_counts, k).mean(),
            "mean_popularity": metrics.mean_popularity_at_k(items, eval_set.item_counts, k).mean(),
            "short_lists": (items[:, k - 1] < 0).mean(),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("k")


def evaluate(
    model: Recommender, eval_set: EvalSet, ks: tuple[int, ...] = config.K_VALUES
) -> pd.DataFrame:
    """Формирует рекомендации обученной модели и считает метрики top-N.

    Args:
        model: Обученная модель.
        eval_set: Набор оценки.
        ks: Длины списков.

    Returns:
        DataFrame с индексом (model, k), колонками метрик и временем
        формирования рекомендаций `seconds`.
    """
    start = time.perf_counter()
    recommendations = model.recommend(eval_set.user_ids, max(ks), exclude=eval_set.seen)
    seconds = time.perf_counter() - start
    result = ranking_metrics(recommendations, eval_set, ks).assign(seconds=seconds)
    return pd.concat({model.name: result}, names=["model"])


def compare(
    models: list[Recommender], eval_set: EvalSet, ks: tuple[int, ...] = config.K_VALUES
) -> pd.DataFrame:
    """Оценивает несколько обученных моделей на одном наборе.

    Args:
        models: Обученные модели.
        eval_set: Набор оценки.
        ks: Длины списков.

    Returns:
        Объединённая таблица `evaluate` по всем моделям.
    """
    return pd.concat([evaluate(model, eval_set, ks) for model in models])


def recommend_all(
    models: dict[str, Recommender], eval_set: EvalSet, k: int = max(config.K_VALUES)
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Формирует top-K списки нескольких обученных моделей для набора оценки.

    Args:
        models: Обученные модели по отображаемому названию.
        eval_set: Набор оценки.
        k: Длина списков.

    Returns:
        Кортеж словарей по названию модели: списки `book_id` формы
        (len(user_ids), K) и время их формирования в секундах.
    """
    recommendations, seconds = {}, {}
    for label, model in models.items():
        start = time.perf_counter()
        recommendations[label] = model.recommend(eval_set.user_ids, k, exclude=eval_set.seen)
        seconds[label] = time.perf_counter() - start
    return recommendations, seconds


def compare_recommendations(
    recommendations: dict[str, np.ndarray],
    eval_set: EvalSet,
    ks: tuple[int, ...] = config.K_VALUES,
) -> pd.DataFrame:
    """Считает метрики top-N для готовых списков нескольких моделей.

    Args:
        recommendations: Списки `book_id` по имени модели, строки выровнены
            с `eval_set.user_ids`.
        eval_set: Набор оценки.
        ks: Длины списков.

    Returns:
        DataFrame с индексом (model, k) и колонками метрик.
    """
    return pd.concat(
        {name: ranking_metrics(recs, eval_set, ks) for name, recs in recommendations.items()},
        names=["model"],
    )


def align_rows(recommendations: np.ndarray, source: EvalSet, target: EvalSet) -> np.ndarray:
    """Выбирает строки списков, построенных для `source`, в порядке пользователей `target`.

    Args:
        recommendations: Списки, строки выровнены с `source.user_ids`.
        source: Набор оценки, для которого построены списки.
        target: Набор оценки, пользователи которого — подмножество `source`.

    Returns:
        Списки, строки выровнены с `target.user_ids`.

    Raises:
        ValueError: Если в `source` нет части пользователей `target`.
    """
    rows = np.searchsorted(source.user_ids, target.user_ids)
    rows = np.minimum(rows, source.user_ids.size - 1)
    if not np.array_equal(source.user_ids[rows], target.user_ids):
        raise ValueError("Пользователи target должны входить в source.")
    return recommendations[rows]


def subset(eval_set: EvalSet, user_ids: np.ndarray) -> EvalSet:
    """Ограничивает набор оценки подмножеством пользователей.

    Args:
        eval_set: Исходный набор оценки.
        user_ids: Идентификаторы пользователей из `eval_set.user_ids`.

    Returns:
        Набор оценки с теми же `seen` и `item_counts`.
    """
    user_ids = np.sort(np.asarray(user_ids))
    rows = np.searchsorted(eval_set.user_ids, user_ids)
    return EvalSet(user_ids, eval_set.relevant[rows], eval_set.seen, eval_set.item_counts)


def user_metrics(recommendations: np.ndarray, eval_set: EvalSet, k: int) -> pd.DataFrame:
    """Считает метрики top-K для каждого пользователя набора оценки.

    Args:
        recommendations: Списки `book_id`, строки выровнены с `eval_set.user_ids`.
        eval_set: Набор оценки.
        k: Длина списка.

    Returns:
        DataFrame с индексом `user_id` и колонками precision, recall, ndcg,
        map, hit_rate, novelty, mean_popularity.
    """
    items = np.where(recommendations > 0, recommendations - 1, -1)
    hits = metrics.hit_matrix(items, eval_set.relevant)
    n_relevant = eval_set.n_relevant
    return pd.DataFrame(
        {
            "precision": metrics.precision_at_k(hits, k),
            "recall": metrics.recall_at_k(hits, n_relevant, k),
            "ndcg": metrics.ndcg_at_k(hits, n_relevant, k),
            "map": metrics.average_precision_at_k(hits, n_relevant, k),
            "hit_rate": metrics.hit_rate_at_k(hits, k),
            "novelty": metrics.novelty_at_k(items, eval_set.item_counts, k),
            "mean_popularity": metrics.mean_popularity_at_k(items, eval_set.item_counts, k),
        },
        index=pd.Index(eval_set.user_ids, name="user_id"),
    )


def segment_metrics(
    recommendations: dict[str, np.ndarray],
    eval_set: EvalSet,
    segments: pd.Series,
    k: int = 10,
    columns: tuple[str, ...] = ("ndcg", "recall", "hit_rate"),
) -> pd.DataFrame:
    """Усредняет метрики top-K по сегментам пользователей.

    Args:
        recommendations: Списки `book_id` по имени модели.
        eval_set: Набор оценки.
        segments: Сегмент пользователя, индекс — `user_id`.
        k: Длина списка.
        columns: Усредняемые метрики.

    Returns:
        DataFrame с индексом (model, segment), колонками метрик, полушириной
        95 % доверительного интервала каждой метрики (`<метрика>_ci`), числом
        пользователей `n_users` и средним числом релевантных книг `n_relevant`.
    """
    user_segment = segments.reindex(eval_set.user_ids).array
    tables = {}
    for name, recs in recommendations.items():
        per_user = user_metrics(recs, eval_set, k)[list(columns)]
        grouped = per_user.assign(n_relevant=eval_set.n_relevant).groupby(
            user_segment, observed=False
        )
        table = grouped[list(columns)].mean()
        n_users = grouped.size()
        for column in columns:
            table[f"{column}_ci"] = 1.96 * grouped[column].std() / np.sqrt(n_users)
        table["n_users"] = n_users
        table["n_relevant"] = grouped["n_relevant"].mean()
        tables[name] = table.rename_axis("segment")
    return pd.concat(tables, names=["model"])


def exposure(
    recommendations: dict[str, np.ndarray], item_segments: pd.Series, k: int = 10
) -> pd.DataFrame:
    """Считает долю позиций top-K, занятых книгами каждого сегмента популярности.

    Args:
        recommendations: Списки `book_id` по имени модели.
        item_segments: Categorical-сегмент книги (head / mid / tail), индекс — `book_id`.
        k: Длина списка.

    Returns:
        DataFrame: строки — модели, колонки — сегменты, значения — доли
        заполненных позиций top-K.
    """
    lookup = item_segments.reindex(np.arange(1, config.N_BOOKS + 1))
    rows = {}
    for name, recs in recommendations.items():
        top = recs[:, :k]
        segment = lookup.to_numpy()[top[top > 0] - 1]
        counts = pd.Series(segment).value_counts(normalize=True)
        rows[name] = counts.reindex(item_segments.cat.categories, fill_value=0.0)
    return pd.DataFrame(rows).T.rename_axis("model")
