"""Симуляция cold start пользователей и книг на обучающей выборке.

Cold start пользователя моделируется усечением истории: у выбранных пользователей
в train остаются первые n оценок по времени, остальные удаляются, и модели
обучаются на усечённых данных. Cold start книги — удаление всех оценок части
каталога из train: такие книги неизвестны CF-моделям, но имеют метаданные.
"""

import dataclasses

import numpy as np
import pandas as pd

from recsys import config, metrics
from recsys.evaluation import EvalSet


def sample_users(
    user_ids: np.ndarray,
    n_users: int = config.COLD_N_USERS,
    random_state: int = config.RANDOM_STATE,
) -> np.ndarray:
    """Выбирает случайное подмножество пользователей.

    Args:
        user_ids: Идентификаторы-кандидаты.
        n_users: Размер выборки.
        random_state: Seed выборки.

    Returns:
        Отсортированные идентификаторы.
    """
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(user_ids, size=min(n_users, len(user_ids)), replace=False))


def truncate_history(train: pd.DataFrame, user_ids: np.ndarray, n: int) -> pd.DataFrame:
    """Оставляет у выбранных пользователей только первые n оценок train.

    Args:
        train: Обучающие оценки в хронологическом порядке строк.
        user_ids: Пользователи с усекаемой историей.
        n: Длина оставляемой истории.

    Returns:
        Оценки train с усечённой историей выбранных пользователей.
    """
    position = train.groupby("user_id", sort=False).cumcount().to_numpy()
    keep = ~train["user_id"].isin(user_ids).to_numpy() | (position < n)
    return train[keep]


def hide_items(
    train: pd.DataFrame,
    share: float = config.COLD_ITEM_SHARE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Удаляет из train все оценки случайной доли книг каталога.

    Args:
        train: Обучающие оценки.
        share: Доля скрываемых книг.
        random_state: Seed выборки книг.

    Returns:
        Кортеж: оценки train без скрытых книг и отсортированные `book_id` скрытых книг.
    """
    rng = np.random.default_rng(random_state)
    hidden = np.sort(
        rng.choice(
            np.arange(1, config.N_BOOKS + 1), size=int(share * config.N_BOOKS), replace=False
        )
    )
    return train[~train["book_id"].isin(hidden)], hidden


def with_train(eval_set: EvalSet, train: pd.DataFrame) -> EvalSet:
    """Заменяет популярность книг набора оценки на рассчитанную по другой обучающей выборке.

    Исключаемые из выдачи книги (`seen`) и релевантные книги не меняются: цель
    оценки остаётся той же, что и без симуляции.

    Args:
        eval_set: Исходный набор оценки.
        train: Обучающие оценки симуляции.

    Returns:
        Набор оценки с `item_counts` по `train`.
    """
    counts = np.bincount(train["book_id"].to_numpy() - 1, minlength=config.N_BOOKS)
    return dataclasses.replace(eval_set, item_counts=counts)


def new_item_metrics(
    recommendations: np.ndarray, eval_set: EvalSet, new_books: np.ndarray, k: int = 10
) -> pd.Series:
    """Считает качество рекомендаций скрытых (новых) книг.

    Args:
        recommendations: Списки `book_id`, строки выровнены с `eval_set.user_ids`.
        eval_set: Набор оценки.
        new_books: `book_id` новых книг.
        k: Длина списка.

    Returns:
        Series: `exposure` — доля позиций top-K с новыми книгами; `recall_new` —
        средний Recall@K по новым книгам среди пользователей, у которых есть
        релевантные новые книги; `hit_rate_new` — доля таких пользователей
        с попаданием в новую книгу; `users_with_new` — их число.
    """
    top = recommendations[:, :k]
    is_new = np.isin(top, new_books)
    new_relevant = eval_set.relevant[:, new_books - 1]
    n_new_relevant = np.diff(new_relevant.indptr)
    users = n_new_relevant > 0
    items = np.where(is_new, top - 1, -1)
    hits = metrics.hit_matrix(items[users], eval_set.relevant[users])
    return pd.Series(
        {
            "exposure": is_new.sum() / max((top > 0).sum(), 1),
            "recall_new": metrics.recall_at_k(hits, n_new_relevant[users], k).mean(),
            "hit_rate_new": metrics.hit_rate_at_k(hits, k).mean(),
            "users_with_new": int(users.sum()),
        }
    )
