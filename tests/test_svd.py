"""Проверка FunkSVD, PureSVD и бейзлайна BaselineOnly: сверка с surprise и numpy."""

import numpy as np
import pandas as pd
from surprise import BaselineOnly

from recsys import config
from recsys.data import interaction_matrix
from recsys.models.svd import FunkSVD, PureSVD, baseline_predict, build_trainset


def _random_train(seed: int = 0, n_users: int = 40, n_books: int = 12) -> pd.DataFrame:
    """Случайные оценки: каждый пользователь оценивает от 3 до 8 книг; книга 12 без оценок."""
    rng = np.random.default_rng(seed)
    rows = []
    for user in range(1, n_users + 1):
        for book in rng.choice(n_books - 1, size=rng.integers(3, 9), replace=False):
            rows.append((user, book + 1, int(rng.integers(1, 6))))
    return pd.DataFrame(rows, columns=["user_id", "book_id", "rating"])


def _model(train: pd.DataFrame) -> FunkSVD:
    """Малая модель для игрушечных данных."""
    return FunkSVD(n_factors=4, n_epochs=10, lr_all=0.01, reg_all=0.05).fit(train)


def test_predict_matches_surprise() -> None:
    train = _random_train()
    model = _model(train)
    users = np.array([1, 5, 17, 40, 41, 3])
    books = np.array([2, 7, 11, 1, 3, 12])  # пользователь 41 и книга 12 отсутствуют в train
    expected = [model.algo.predict(int(u), int(b)).est for u, b in zip(users, books, strict=True)]
    np.testing.assert_allclose(model.predict(users, books), expected, atol=1e-5)


def test_score_consistent_with_predict() -> None:
    train = _random_train()
    model = _model(train)
    users = np.array([2, 9])
    scores = model.score(users)
    assert scores.shape == (2, config.N_BOOKS)
    assert np.isneginf(scores[:, 11:]).all()  # книги без оценок в train
    books = np.arange(1, 12)
    for row, user in enumerate(users):
        predicted = model.predict(np.full(books.size, user), books)
        clipped = np.clip(scores[row, :11], config.RATING_MIN, config.RATING_MAX)
        np.testing.assert_allclose(clipped, predicted, atol=1e-5)


def test_recommend_excludes_seen_and_sorts() -> None:
    train = _random_train()
    model = _model(train)
    seen = interaction_matrix(train, binary=True)
    user = 3
    top = model.recommend(np.array([user]), 3, exclude=seen)[0]
    rated = set(train.loc[train["user_id"] == user, "book_id"])
    assert not rated & set(top.tolist())
    candidates = np.array([b for b in range(1, 12) if b not in rated])
    scores = model.score(np.array([user]))[0, candidates - 1]
    np.testing.assert_array_equal(top, candidates[np.argsort(-scores, kind="stable")[:3]])


def test_fit_is_deterministic() -> None:
    train = _random_train()
    first, second = _model(train), _model(train)
    np.testing.assert_array_equal(first.item_factors, second.item_factors)
    np.testing.assert_array_equal(first.user_bias, second.user_bias)


def test_get_recommendations_columns() -> None:
    train = _random_train()
    model = _model(train)
    books = pd.DataFrame(
        {
            "book_id": np.arange(1, config.N_BOOKS + 1),
            "title": [f"t{i}" for i in range(config.N_BOOKS)],
            "authors": "a",
        }
    )
    result = model.get_recommendations(1, books, n=5)
    assert list(result.columns) == ["book_id", "title", "authors", "predicted_rating"]
    assert result["predicted_rating"].is_monotonic_decreasing


def test_min_item_ratings_filters_candidates() -> None:
    train = _random_train()
    model = _model(train)
    counts = train["book_id"].value_counts()
    threshold = int(counts.median())
    model.set_min_item_ratings(threshold)
    scores = model.score(np.array([1]))[0, :11]
    rare = counts[counts < threshold].index.to_numpy()
    assert np.isneginf(scores[rare - 1]).all()
    assert np.isfinite(np.delete(scores, rare - 1)).all()


def test_baseline_predict_matches_surprise() -> None:
    train = _random_train()
    users = np.array([1, 8, 40, 41])
    books = np.array([3, 5, 12, 1])
    algo = BaselineOnly(verbose=False).fit(build_trainset(train))
    expected = [algo.predict(int(u), int(b)).est for u, b in zip(users, books, strict=True)]
    np.testing.assert_allclose(baseline_predict(train, users, books), expected, atol=1e-5)


def test_pure_svd_matches_dense_svd() -> None:
    train = _random_train()
    k = 3
    model = PureSVD(n_factors=k).fit(train)
    # Нулевые строки и столбцы не меняют сингулярные числа и V.
    dense = interaction_matrix(train)[:40, :12].toarray()
    _, singular_values, vt = np.linalg.svd(dense, full_matrices=False)
    np.testing.assert_allclose(model.singular_values, singular_values[:k], rtol=1e-4)
    users = np.array([1, 7])
    expected = dense[users - 1] @ vt[:k].T @ vt[:k]
    scores = model.score(users)
    np.testing.assert_allclose(scores[:, :11], expected[:, :11], atol=1e-4)
    assert np.isneginf(scores[:, 11:]).all()
