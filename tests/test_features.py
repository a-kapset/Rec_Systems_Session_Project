"""Проверка признаков: категории книг, агрегаты книг и пользователей, признаки пар."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from recsys import config, features, split

PATTERNS = {"fantasy": r"fantasy|magic", "romance": r"romance|^love$", "horror": r"horror"}


def _categories() -> pd.Series:
    """Книги 1–2 — fantasy, 3 — romance, остальные — без совпадений (other)."""
    book_tags = pd.DataFrame(
        {
            "book_id": [1, 1, 2, 3, 3, 4],
            "tag_name": ["fantasy", "love", "magic-romance", "romance", "magic", "fiction"],
            "count": [10, 5, 8, 7, 3, 9],
        }
    )
    return features.book_categories(book_tags, PATTERNS)


def test_tag_category_matrix_allows_several_categories() -> None:
    matches = features.tag_category_matrix(
        pd.Series(["paranormal-romance", "magic", "fiction", "love", "lovely"]), PATTERNS
    )
    assert matches.loc["paranormal-romance"].tolist() == [False, True, False]
    assert matches.loc["magic", "fantasy"]
    assert not matches.loc["fiction"].any()
    assert matches.loc["love", "romance"]
    assert not matches.loc["lovely"].any()


def test_book_categories_by_max_count() -> None:
    categories = _categories()
    assert list(categories.cat.categories) == [*PATTERNS, config.CATEGORY_OTHER]
    assert len(categories) == config.N_BOOKS
    # Книга 1: fantasy 10 > romance 5; книга 2: равенство fantasy = romance = 8 → первая;
    # книга 3: romance 7 > fantasy 3; книга 4 и книги без тегов — other.
    assert categories.loc[[1, 2, 3, 4, 5]].tolist() == [
        "fantasy",
        "fantasy",
        "romance",
        config.CATEGORY_OTHER,
        config.CATEGORY_OTHER,
    ]


def _ratings() -> pd.DataFrame:
    """Пользователь 1 — 5 оценок, пользователь 2 — 5 оценок, пользователь 3 — 2 оценки."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3],
            "book_id": [1, 2, 3, 4, 5, 1, 3, 4, 5, 2, 1, 3],
            "rating": [5, 4, 2, 1, 5, 3, 5, 4, 4, 1, 4, 5],
        }
    )


def test_book_features_use_only_train() -> None:
    ratings = _ratings()
    train = split.temporal_split(ratings).train
    result = features.book_features(train, _categories(), bayes_m=2)
    # Последние оценки пользователей 1 и 2 (книги 5 и 2) ушли в test.
    book_1 = train[train["book_id"] == 1]["rating"]
    assert result.loc[1, "n_ratings"] == 3
    assert result.loc[1, "mean_rating"] == pytest.approx(book_1.mean())
    assert result.loc[1, "std_rating"] == pytest.approx(book_1.std(ddof=0))
    assert result.loc[2, "n_ratings"] == 1
    assert result.loc[2, "low_share"] == 0.0
    assert result.loc[4, "low_share"] == pytest.approx(0.5)
    global_mean = train["rating"].mean()
    assert result.loc[1, "bayes_rating"] == pytest.approx((book_1.sum() + 2 * global_mean) / 5)
    assert result.loc[6, "n_ratings"] == 0
    assert result.loc[6, "bayes_rating"] == pytest.approx(global_mean)
    assert np.isnan(result.loc[6, "mean_rating"])
    assert result.loc[3, "category"] == "romance"


def test_user_features_aggregates_and_type() -> None:
    ratings = _ratings()
    train = split.temporal_split(ratings).train
    result = features.user_features(train, _categories(), new_max_ratings=3)
    user_1 = train[train["user_id"] == 1]["rating"]
    assert result.loc[1, "n_ratings"] == 4
    assert result.loc[1, "mean_rating"] == pytest.approx(user_1.mean())
    assert result.loc[1, "std_rating"] == pytest.approx(user_1.std(ddof=0))
    assert result.loc[1, "liked_share"] == pytest.approx(0.5)
    # Пользователь 1 в train: книги 1–4; категории fantasy (1, 2), romance (3), other (4).
    assert result.loc[1, "n_categories"] == 3
    book_n = train.groupby("book_id").size()
    expected = np.log1p(book_n.loc[[1, 2, 3, 4]]).mean()
    assert result.loc[1, "mainstream"] == pytest.approx(expected)
    assert result.loc[1, "user_type"] == "active"
    assert result.loc[3, "user_type"] == "new"
    assert result.loc[4, "n_ratings"] == 0
    assert result.loc[4, "user_type"] == "new"


def test_category_profile_and_affinity() -> None:
    train = _ratings()
    categories = _categories()
    profile = features.category_profile(train, categories)
    # Понравившиеся книги пользователя 1: 1, 2 (fantasy), 5 (other).
    assert profile.loc[1, "fantasy"] == pytest.approx(2 / 3)
    assert profile.loc[1, config.CATEGORY_OTHER] == pytest.approx(1 / 3)
    assert profile.loc[4].sum() == 0.0
    affinity = features.category_affinity(train, categories, alpha=2.0)
    liked = train[train["rating"] >= config.RELEVANCE_THRESHOLD]
    prior_fantasy = categories.loc[liked["book_id"]].eq("fantasy").mean()
    assert affinity[0, 0] == pytest.approx((2 + 2 * prior_fantasy) / (3 + 2))
    assert affinity[3, 0] == pytest.approx(prior_fantasy)
    np.testing.assert_allclose(affinity.sum(axis=1), 1.0, rtol=1e-6)


def test_gather_scores_matches_full_matrix() -> None:
    rng = np.random.default_rng(0)
    table = rng.random((10, config.N_BOOKS)).astype(np.float32)

    def score_fn(user_ids: np.ndarray) -> np.ndarray:
        return table[user_ids - 1]

    user_ids = np.array([3, 1, 3, 10, 7, 1])
    book_ids = np.array([5, 1, 9_999, 2, 10_000, 3])
    result = features.gather_scores(score_fn, user_ids, book_ids, batch_size=2)
    np.testing.assert_array_equal(result, table[user_ids - 1, book_ids - 1])


def test_sample_unseen_books_avoids_excluded() -> None:
    allowed = np.array([7, 42])
    mask = np.ones(config.N_BOOKS, dtype=np.float32)
    mask[allowed - 1] = 0.0
    exclude = sparse.csr_matrix(np.tile(mask, (3, 1)))
    exclude.resize((config.N_USERS, config.N_BOOKS))
    sampled = features.sample_unseen_books(np.array([1, 1, 2, 3, 3, 3]), exclude, random_state=0)
    assert np.isin(sampled, allowed).all()
