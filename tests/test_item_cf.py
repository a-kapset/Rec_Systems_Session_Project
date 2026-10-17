"""Проверка Item-based CF: схожесть, отбор соседей, предсказание оценки и top-N."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from recsys.data import interaction_matrix
from recsys.models.item_cf import ItemCF, item_similarity, prune_top_k


def _random_train(seed: int = 0, n_users: int = 30, n_books: int = 8) -> pd.DataFrame:
    """Случайные оценки: каждый пользователь оценивает от 3 до 6 книг."""
    rng = np.random.default_rng(seed)
    rows = []
    for user in range(1, n_users + 1):
        for book in rng.choice(n_books, size=rng.integers(3, 7), replace=False):
            rows.append((user, book + 1, int(rng.integers(1, 6))))
    return pd.DataFrame(rows, columns=["user_id", "book_id", "rating"])


def _reference_similarity(
    train: pd.DataFrame, n_books: int, shrinkage: float, positive_only: bool = True
) -> np.ndarray:
    """Adjusted cosine с shrinkage прямым расчётом по плотной матрице."""
    dense = train.pivot_table(index="user_id", columns="book_id", values="rating").reindex(
        columns=range(1, n_books + 1)
    )
    observed = dense.notna().to_numpy()
    centered = dense.sub(dense.mean(axis=1), axis=0).fillna(0.0).to_numpy()
    norms = np.linalg.norm(centered, axis=0)
    sim = centered.T @ centered / np.outer(norms, norms)
    co_counts = observed.T.astype(float) @ observed.astype(float)
    sim *= co_counts / (co_counts + shrinkage)
    np.fill_diagonal(sim, 0.0)
    sim = np.nan_to_num(sim)
    return np.clip(sim, 0.0, None) if positive_only else sim


def test_similarity_matches_reference() -> None:
    train = _random_train()
    expected = _reference_similarity(train, 8, shrinkage=5.0)
    result = item_similarity(interaction_matrix(train), "adjusted_cosine", 5.0).toarray()
    np.testing.assert_allclose(result[:8, :8], expected, atol=1e-5)
    assert not result[8:].any()


def test_cosine_similarity_manual() -> None:
    # Книга 1: (5, 3), книга 2: (4, 0), общий пользователь один.
    train = pd.DataFrame({"user_id": [1, 1, 2], "book_id": [1, 2, 1], "rating": [5, 4, 3]})
    sim = item_similarity(interaction_matrix(train), "cosine").toarray()
    assert sim[0, 1] == pytest.approx(5 * 4 / (np.sqrt(34) * 4))
    assert sim[0, 0] == 0.0
    shrunk = item_similarity(interaction_matrix(train), "cosine", shrinkage=1.0).toarray()
    assert shrunk[0, 1] == pytest.approx(sim[0, 1] / 2)


def test_prune_top_k() -> None:
    matrix = sparse.csr_matrix(np.array([[0.0, 0.2, 0.9, 0.5], [0.3, 0.0, 0.0, 0.0]]))
    pruned = prune_top_k(matrix, 2).toarray()
    np.testing.assert_allclose(pruned, [[0.0, 0.0, 0.9, 0.5], [0.3, 0.0, 0.0, 0.0]])


@pytest.mark.parametrize(
    ("centered", "use_negative"), [(True, False), (False, False), (True, True), (False, True)]
)
def test_predict_matches_reference(centered: bool, use_negative: bool) -> None:
    train = _random_train(seed=1)
    k = 2
    model = ItemCF(
        "adjusted_cosine", k, shrinkage=5.0, centered=centered, use_negative=use_negative
    ).fit(train)
    sim = _reference_similarity(train, 8, shrinkage=5.0, positive_only=not use_negative)
    user_means = train.groupby("user_id")["rating"].mean()
    users = np.repeat(np.arange(1, 31), 8)
    books = np.tile(np.arange(1, 9), 30)
    expected = []
    for user, book in zip(users, books, strict=True):
        history = train[train["user_id"] == user]
        sims = sim[book - 1, history["book_id"].to_numpy() - 1]
        values = history["rating"].to_numpy() - (user_means[user] if centered else 0.0)
        top = np.argsort(-np.abs(sims), kind="stable")[:k]
        weight = np.abs(sims[top]).sum()
        if weight == 0:
            expected.append(user_means[user])
            continue
        estimate = (sims[top] * values[top]).sum() / weight
        expected.append(estimate + (user_means[user] if centered else 0.0))
    expected = np.clip(expected, 1, 5)
    np.testing.assert_allclose(model.predict(users, books), expected, atol=1e-4)


def test_predict_fallback() -> None:
    train = _random_train(seed=2)
    model = ItemCF(n_neighbors=5).fit(train)
    # Пользователь 100 не имеет оценок: r̄_i; книга 50 не имеет оценок: r̄_u.
    prediction = model.predict(np.array([100, 1]), np.array([1, 50]))
    item_mean = train.loc[train["book_id"] == 1, "rating"].mean()
    user_mean = train.loc[train["user_id"] == 1, "rating"].mean()
    np.testing.assert_allclose(prediction, [item_mean, user_mean], atol=1e-5)


def test_score_similarity_ranking() -> None:
    train = _random_train(seed=3)
    model = ItemCF(n_neighbors=3, shrinkage=0.0, ranking="similarity").fit(train)
    liked = interaction_matrix(train[train["rating"] >= 4], binary=True).toarray()
    expected = liked[:5] @ model.neighbors.toarray().T
    np.testing.assert_allclose(model.score(np.arange(1, 6)), expected, atol=1e-5)
