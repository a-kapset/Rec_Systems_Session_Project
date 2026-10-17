"""Проверка гибрида: нормализация, веса по типам пользователей, fallback, полнота выдачи."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from test_candidates import FixedScores

from recsys import config, evaluation, hybrid
from recsys.balancing import Quotas


def test_normalize_per_user() -> None:
    values = np.array([[1, -np.inf], [2, 0], [3, 2], [5, 1], [5, 3]], dtype=np.float32)
    offsets = np.array([0, 3, 3, 5])  # второй пользователь без кандидатов
    expected = [[0, 0], [0.5, 0], [1, 1], [0, 0], [0, 1]]
    np.testing.assert_allclose(hybrid.normalize_per_user(values, offsets), expected)


def test_weight_matrix() -> None:
    matrix = hybrid.weight_matrix({"new": {"b": 1.0}, "active": {"a": 0.5}}, ("a", "b"))
    np.testing.assert_array_equal(matrix, [[0, 1], [0.5, 0]])
    with pytest.raises(ValueError):
        hybrid.weight_matrix({"active": {"c": 1.0}}, ("a", "b"))


def test_fill_fallback() -> None:
    recommendations = np.array([[5, -1, -1], [7, 8, 9]])
    seen = sparse.csr_matrix(([1.0], ([0], [0])), shape=(config.N_USERS, config.N_BOOKS))
    order = np.arange(10)  # book_id 1, 2, ...
    result = hybrid.fill_fallback(recommendations, np.array([1, 2]), order, seen)
    # Книга 1 прочитана пользователем 1, книга 5 уже в списке.
    np.testing.assert_array_equal(result, [[5, 2, 3], [7, 8, 9]])


@pytest.fixture
def toy() -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    """Пользователь 1 — 12 оценок (active), 2 — 2 оценки (new), 3 — без оценок."""
    train = pd.DataFrame(
        {
            "user_id": [1] * 12 + [2, 2],
            "book_id": [*range(1, 13), 1, 2],
            "rating": [5, 4, 5, 3, 4, 5, 2, 4, 5, 4, 3, 5, 5, 4],
        }
    )
    # Категории пар книг чередуются: книги 13–14 — a, 15–16 — b, 17–18 — c, 19–20 — a.
    codes = (np.arange(config.N_BOOKS) // 2) % 3
    categories = pd.Series(
        pd.Categorical.from_codes(codes, ["a", "b", "c"]),
        index=pd.RangeIndex(1, config.N_BOOKS + 1, name="book_id"),
    )
    # Popularity: одинаковые убывающие скоры книг 1–30 у всех пользователей;
    # Item-CF: убывающие скоры книг 13–40 у пользователей 1 и 2, у пользователя 3 — нули.
    popularity = np.zeros((3, config.N_BOOKS), dtype=np.float32)
    popularity[:, :30] = np.arange(30, 0, -1)
    item_cf = np.zeros((3, config.N_BOOKS), dtype=np.float32)
    item_cf[:2, 12:40] = np.linspace(1.0, 0.1, 28)
    models = {
        "popularity": FixedScores(popularity, "popularity"),
        "item_cf": FixedScores(item_cf, "item_cf"),
    }
    return train, categories, models


WEIGHTS = {"active": {"item_cf": 1.0}, "new": {"popularity": 1.0}}


def test_hybrid_user_types(toy: tuple[pd.DataFrame, pd.Series, dict]) -> None:
    train, categories, models = toy
    model = hybrid.HybridRecommender(
        models, categories, sources=("popularity", "item_cf"), weights=WEIGHTS, n_candidates=5
    ).fit(train)
    np.testing.assert_array_equal(model.user_types(np.array([1, 2, 3])), [1, 0, 0])
    recs = model.recommend(np.array([1, 2, 3]), k=4)
    # Активный пользователь — порядок Item-CF; новые — порядок Popularity без прочитанного.
    np.testing.assert_array_equal(recs[0], [13, 14, 15, 16])
    np.testing.assert_array_equal(recs[1], [3, 4, 5, 6])
    np.testing.assert_array_equal(recs[2], [1, 2, 3, 4])
    seen = model.seen.toarray()
    for row, user in enumerate([1, 2, 3]):
        assert not seen[user - 1, recs[row] - 1].any()


def test_hybrid_empty_pool_uses_popularity(toy: tuple[pd.DataFrame, pd.Series, dict]) -> None:
    train, categories, models = toy
    model = hybrid.HybridRecommender(
        models, categories, sources=("item_cf",), weights=WEIGHTS, n_candidates=3
    ).fit(train)
    recs = model.recommend(np.array([1, 3]), k=5)
    # Пользователь 1: 3 кандидата Item-CF и 2 книги Popularity вне train и пула;
    # пользователь 3: пул пуст, весь список — Popularity.
    np.testing.assert_array_equal(recs, [[13, 14, 15, 16, 17], [1, 2, 3, 4, 5]])


def test_hybrid_quotas_and_score(toy: tuple[pd.DataFrame, pd.Series, dict]) -> None:
    train, categories, models = toy
    base = hybrid.HybridRecommender(models, categories, sources=("item_cf",), weights=WEIGHTS)
    base.fit(train)
    balanced = base.with_params(quotas=Quotas(max_per_category=1), name="Hybrid+q")
    np.testing.assert_array_equal(base.recommend(np.array([1]), k=3), [[13, 14, 15]])
    np.testing.assert_array_equal(balanced.recommend(np.array([1]), k=3), [[13, 15, 17]])
    assert base.quotas is None
    assert np.isfinite(base.score(np.array([1]))).sum() == 28
    with pytest.raises(ValueError):
        base.with_params(n_candidates=3)


def test_search_weights_and_quotas(toy: tuple[pd.DataFrame, pd.Series, dict]) -> None:
    train, categories, models = toy
    model = hybrid.HybridRecommender(
        models, categories, sources=("popularity", "item_cf"), weights=WEIGHTS, n_candidates=5
    ).fit(train)
    # Новому пользователю 2 релевантны книги 3 и 4 — первые в порядке Popularity
    # (у Item-CF — книги 13–17); пользователю 1 (active) — книга 13.
    holdout = pd.DataFrame({"user_id": [1, 2, 2], "book_id": [13, 3, 4], "rating": [5, 4, 5]})
    eval_set = evaluation.build_eval_set(train, holdout, n_users=None)
    scored = model.prepare(eval_set.user_ids)
    table = hybrid.search_weights(
        model, scored, eval_set, "new", ("popularity", "item_cf"), values=(0.0, 1.0), k=2
    )
    assert len(table) == 3
    assert table.loc[0, ["popularity", "item_cf"]].tolist() == [1.0, 0.0]
    assert table.loc[0, "ndcg"] == pytest.approx(1.0)
    weights = hybrid.best_weights(model, table, "new")
    assert weights == {"active": {"item_cf": 1.0}, "new": {"popularity": 1.0}}
    assert hybrid.normalize_weights({"new": {"a": 1.0, "b": 3.0, "c": 0.0}}) == {
        "new": {"a": 0.25, "b": 0.75}
    }
    variants = hybrid.quota_grid((1, 2), (0,))
    quotas_table = hybrid.search_quotas(model, scored, eval_set, variants, k=2)
    assert list(quotas_table.index) == ["без квот", "q=1", "q=2"]
    selected = hybrid.select_quotas(quotas_table, variants, max_drop=1.0)
    assert selected in variants
