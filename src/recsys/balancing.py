"""Балансировка релевантности и разнообразия выдачи квотами.

Жадное переранжирование: книги пула просматриваются по убыванию гибридного скора
и принимаются в top-K, если не нарушают квот — не более q книг одной категории
и резерв позиций под книги вне head популярности (не менее t). Если квоты
невыполнимы из-за состава пула, они ослабляются по очереди: сначала снимается
квота long tail, затем квота категорий. Выбранные книги упорядочиваются по скору.
Обработка векторизована по пользователям, цикл — по позициям ранжированного пула.
"""

from dataclasses import dataclass

import numpy as np

from recsys import config


@dataclass(frozen=True)
class Quotas:
    """Квоты балансировки top-K.

    Квоты относятся ко всему списку длины K: для оценки на нескольких K
    сбалансированный список строится отдельно для каждого K, а не усекается.

    Attributes:
        max_per_category: q — максимум книг одной категории; None — без ограничения.
        min_tail: t — минимум книг вне head популярности.
    """

    max_per_category: int | None = config.BALANCE_MAX_PER_CATEGORY
    min_tail: int = 0

    @property
    def label(self) -> str:
        """Краткое описание квот."""
        parts = []
        if self.max_per_category is not None:
            parts.append(f"q={self.max_per_category}")
        if self.min_tail > 0:
            parts.append(f"t={self.min_tail}")
        return ", ".join(parts) or "без квот"


def head_mask(item_counts: np.ndarray, share: float = config.BALANCE_HEAD_SHARE) -> np.ndarray:
    """Отмечает head — долю `share` самых популярных книг каталога по числу оценок.

    Args:
        item_counts: Число оценок каждой книги в train формы (N_BOOKS,).
        share: Доля каталога, относимая к head.

    Returns:
        Булев массив формы (N_BOOKS,); при равенстве счётчиков — меньший `book_id`.
    """
    n_head = round(share * item_counts.size)
    order = np.argsort(-item_counts, kind="stable")
    mask = np.zeros(item_counts.size, dtype=bool)
    mask[order[:n_head]] = True
    return mask


def rerank(
    ranked: np.ndarray,
    k: int,
    categories: np.ndarray,
    is_head: np.ndarray,
    quotas: Quotas,
) -> np.ndarray:
    """Выбирает top-K из ранжированного пула с соблюдением квот.

    Args:
        ranked: `book_id` формы (n_users, P) по убыванию скора, -1 — пустая позиция.
        k: Длина списка.
        categories: Код категории каждой книги формы (N_BOOKS,).
        is_head: Принадлежность книги head формы (N_BOOKS,).
        quotas: Квоты.

    Returns:
        `book_id` формы (n_users, K); -1 — пустая позиция при пуле короче K.
    """
    n_users, width = ranked.shape
    users = np.arange(n_users)
    chosen = np.full((n_users, k), width, dtype=np.int64)
    filled = np.zeros(n_users, dtype=np.int64)
    used = np.zeros((n_users, width), dtype=bool)
    category_counts = np.zeros((n_users, int(categories.max()) + 1), dtype=np.int64)
    tail_counts = np.zeros(n_users, dtype=np.int64)
    limit = k if quotas.max_per_category is None else quotas.max_per_category
    # Проходы с ослаблением квот: (q, t) → (q, 0) → без квот.
    for pass_limit, min_tail in ((limit, quotas.min_tail), (limit, 0), (k, 0)):
        for position in range(width):
            books = ranked[:, position]
            index = np.where(books > 0, books - 1, 0)
            category, tail = categories[index], ~is_head[index]
            reserve = np.maximum(min_tail - tail_counts, 0)
            accept = (
                (books > 0)
                & ~used[:, position]
                & (filled < k)
                & (category_counts[users, category] < pass_limit)
                & (tail | (k - filled > reserve))
            )
            rows = users[accept]
            chosen[rows, filled[rows]] = position
            filled[rows] += 1
            used[rows, position] = True
            category_counts[rows, category[accept]] += 1
            tail_counts[rows] += tail[accept]
        if (filled >= k).all():
            break
    chosen.sort(axis=1)
    padded = np.concatenate([ranked, np.full((n_users, 1), -1, dtype=ranked.dtype)], axis=1)
    return np.take_along_axis(padded, chosen, axis=1)
