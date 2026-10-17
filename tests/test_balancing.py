"""Проверка балансировки: head популярности, квоты категорий и long tail."""

import numpy as np

from recsys import balancing
from recsys.balancing import Quotas

# Книги 1–5: категории 0, 0, 1, 2, 0; head — книги 1–3.
CATEGORIES = np.array([0, 0, 1, 2, 0])
IS_HEAD = np.array([True, True, True, False, False])


def test_head_mask() -> None:
    counts = np.array([5, 50, 7, 50, 1, 0, 3, 2, 9, 4])
    mask = balancing.head_mask(counts, share=0.2)
    np.testing.assert_array_equal(np.flatnonzero(mask), [1, 3])


def test_category_quota() -> None:
    ranked = np.array([[1, 2, 3, 4, 5]])
    result = balancing.rerank(ranked, 3, CATEGORIES, IS_HEAD, Quotas(max_per_category=1))
    np.testing.assert_array_equal(result, [[1, 3, 4]])


def test_tail_quota() -> None:
    ranked = np.array([[1, 2, 3, 4, 5]])
    result = balancing.rerank(ranked, 3, CATEGORIES, IS_HEAD, Quotas(None, min_tail=2))
    np.testing.assert_array_equal(result, [[1, 4, 5]])


def test_infeasible_quota_is_relaxed() -> None:
    ranked = np.array([[1, 2, 5, -1, -1], [1, -1, -1, -1, -1]])
    result = balancing.rerank(ranked, 3, CATEGORIES, IS_HEAD, Quotas(max_per_category=1))
    np.testing.assert_array_equal(result, [[1, 2, 5], [1, -1, -1]])


def test_tail_quota_relaxed_before_category_quota() -> None:
    # Книг вне head нет: квота t снимается, квота категорий (q = 1) сохраняется.
    ranked = np.array([[1, 2, 3]])
    is_head = np.ones(5, dtype=bool)
    result = balancing.rerank(ranked, 2, CATEGORIES, is_head, Quotas(1, min_tail=2))
    np.testing.assert_array_equal(result, [[1, 3]])


def test_selected_books_keep_score_order() -> None:
    # Вне head только книга 4: квота t = 2 невыполнима, книга 2 добавляется вторым
    # проходом, но в списке стоит по скору — перед книгой 4.
    ranked = np.array([[1, 2, 3, 4]])
    result = balancing.rerank(ranked, 3, CATEGORIES, IS_HEAD, Quotas(None, min_tail=2))
    np.testing.assert_array_equal(result, [[1, 2, 4]])


def test_quota_label() -> None:
    assert Quotas(3, 2).label == "q=3, t=2"
    assert Quotas(None).label == "без квот"


def test_quotas_for_length() -> None:
    assert Quotas(3, 3).for_length(10) == Quotas(3, 3)
    assert Quotas(3, 3).for_length(20) == Quotas(6, 6)
    assert Quotas(3, 3).for_length(5) == Quotas(2, 2)
    assert Quotas(None, 0).for_length(5) == Quotas(None, 0)
