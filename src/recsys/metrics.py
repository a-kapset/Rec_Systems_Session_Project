"""Метрики качества рекомендаций и предсказания оценок.

Метрики ранжирования считаются векторизованно по матрице попаданий
`hits` формы (n_users, K): `hits[u, j]` — релевантна ли книга на позиции j+1
в списке пользователя u. Функции возвращают значения по пользователям;
усреднение (macro) выполняется в `recsys.evaluation`. Пустые позиции списка
кодируются значением -1 и считаются промахами.
"""

import numpy as np
from scipy import sparse


def hit_matrix(recommendations: np.ndarray, relevant: sparse.csr_matrix) -> np.ndarray:
    """Строит матрицу попаданий рекомендаций в релевантные объекты.

    Args:
        recommendations: Индексы объектов формы (n_users, K), -1 — пустая позиция.
        relevant: Бинарная матрица (n_users, n_items) релевантных объектов;
            строки выровнены с `recommendations`.

    Returns:
        Булева матрица формы (n_users, K).
    """
    n_items = relevant.shape[1]
    rows, cols = relevant.nonzero()
    relevant_keys = rows.astype(np.int64) * n_items + cols
    user_index = np.arange(recommendations.shape[0], dtype=np.int64)[:, None]
    rec_keys = user_index * n_items + recommendations
    return np.isin(rec_keys, relevant_keys) & (recommendations >= 0)


def _check_n_relevant(n_relevant: np.ndarray) -> None:
    """Проверяет, что у каждого пользователя есть релевантные объекты.

    Args:
        n_relevant: Число релевантных объектов по пользователям.

    Raises:
        ValueError: Если у пользователя нет релевантных объектов.
    """
    if np.any(n_relevant < 1):
        raise ValueError("у каждого пользователя должен быть хотя бы один релевантный объект")


def precision_at_k(hits: np.ndarray, k: int) -> np.ndarray:
    """Считает Precision@K = |rec@K ∩ rel| / K.

    Args:
        hits: Матрица попаданий (n_users, ≥ K).
        k: Длина списка.

    Returns:
        Значения по пользователям.
    """
    return hits[:, :k].sum(axis=1) / k


def recall_at_k(hits: np.ndarray, n_relevant: np.ndarray, k: int) -> np.ndarray:
    """Считает Recall@K = |rec@K ∩ rel| / |rel|.

    Args:
        hits: Матрица попаданий (n_users, ≥ K).
        n_relevant: Число всех релевантных объектов пользователя.
        k: Длина списка.

    Returns:
        Значения по пользователям.
    """
    _check_n_relevant(n_relevant)
    return hits[:, :k].sum(axis=1) / n_relevant


def ndcg_at_k(hits: np.ndarray, n_relevant: np.ndarray, k: int) -> np.ndarray:
    """Считает nDCG@K с бинарным gain.

    DCG@K = Σ rel_i / log2(i + 1); IDCG@K — DCG идеального списка, в котором
    все min(K, |rel|) первых позиций заняты релевантными объектами.

    Args:
        hits: Матрица попаданий (n_users, ≥ K).
        n_relevant: Число всех релевантных объектов пользователя.
        k: Длина списка.

    Returns:
        Значения по пользователям.
    """
    _check_n_relevant(n_relevant)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = hits[:, :k] @ discounts
    idcg = np.cumsum(discounts)[np.minimum(n_relevant, k) - 1]
    return dcg / idcg


def average_precision_at_k(hits: np.ndarray, n_relevant: np.ndarray, k: int) -> np.ndarray:
    """Считает AP@K = (1 / min(K, |rel|)) · Σ P@i · rel_i.

    Args:
        hits: Матрица попаданий (n_users, ≥ K).
        n_relevant: Число всех релевантных объектов пользователя.
        k: Длина списка.

    Returns:
        Значения по пользователям; среднее по пользователям — MAP@K.
    """
    _check_n_relevant(n_relevant)
    top = hits[:, :k]
    precision_at_i = np.cumsum(top, axis=1) / np.arange(1, k + 1)
    return (precision_at_i * top).sum(axis=1) / np.minimum(n_relevant, k)


def hit_rate_at_k(hits: np.ndarray, k: int) -> np.ndarray:
    """Считает HitRate@K: наличие хотя бы одного попадания в top-K.

    Args:
        hits: Матрица попаданий (n_users, ≥ K).
        k: Длина списка.

    Returns:
        Значения 0/1 по пользователям.
    """
    return hits[:, :k].any(axis=1).astype(np.float64)


def coverage_at_k(recommendations: np.ndarray, n_catalog: int, k: int) -> float:
    """Считает Coverage@K — долю каталога, попавшую хотя бы в один top-K.

    Args:
        recommendations: Индексы объектов (n_users, ≥ K), -1 — пустая позиция.
        n_catalog: Размер каталога, доступного модели (объекты из train).
        k: Длина списка.

    Returns:
        Доля каталога.
    """
    top = recommendations[:, :k]
    return np.unique(top[top >= 0]).size / n_catalog


def novelty_at_k(recommendations: np.ndarray, item_counts: np.ndarray, k: int) -> np.ndarray:
    """Считает Novelty@K — среднее −log2 p(i) по списку.

    p(i) — доля оценок объекта в train со сглаживанием Лапласа
    (n_i + 1) / (Σ n + n_items), чтобы объекты без оценок имели конечную новизну.

    Args:
        recommendations: Индексы объектов (n_users, ≥ K), -1 — пустая позиция.
        item_counts: Число оценок каждого объекта в train.
        k: Длина списка.

    Returns:
        Значения по пользователям (пустые позиции не учитываются).
    """
    probability = (item_counts + 1.0) / (item_counts.sum() + item_counts.size)
    self_information = -np.log2(probability)
    top = recommendations[:, :k]
    valid = top >= 0
    values = np.where(valid, self_information[np.where(valid, top, 0)], 0.0)
    return values.sum(axis=1) / np.maximum(valid.sum(axis=1), 1)


def mean_popularity_at_k(
    recommendations: np.ndarray, item_counts: np.ndarray, k: int
) -> np.ndarray:
    """Считает среднее число оценок в train у книг из top-K.

    Args:
        recommendations: Индексы объектов (n_users, ≥ K), -1 — пустая позиция.
        item_counts: Число оценок каждого объекта в train.
        k: Длина списка.

    Returns:
        Значения по пользователям (пустые позиции не учитываются).
    """
    top = recommendations[:, :k]
    valid = top >= 0
    values = np.where(valid, item_counts[np.where(valid, top, 0)], 0.0)
    return values.sum(axis=1) / np.maximum(valid.sum(axis=1), 1)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Считает RMSE предсказанных оценок.

    Args:
        y_true: Фактические оценки.
        y_pred: Предсказанные оценки.

    Returns:
        Корень из средней квадратичной ошибки.
    """
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Считает MAE предсказанных оценок.

    Args:
        y_true: Фактические оценки.
        y_pred: Предсказанные оценки.

    Returns:
        Средняя абсолютная ошибка.
    """
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
