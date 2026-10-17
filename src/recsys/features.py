"""Расширенные признаки книг, пользователей и пар «пользователь × книга».

Тематическая категория книги — категория с наибольшей суммой счётчиков очищенных
тегов, сопоставленных категориям по шаблонам ключевых слов (`config.BOOK_CATEGORIES`).
Агрегаты книг и пользователей считаются только по переданной обучающей выборке
(train или train_inner) через `np.bincount` и разреженные матрицы; признаки пар
вычисляются по обученным моделям батчами пользователей.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import sparse

from recsys import config
from recsys.models import ContentBased, FunkSVD, ItemCF


def tag_category_matrix(
    tag_names: pd.Series, patterns: dict[str, str] = config.BOOK_CATEGORIES
) -> pd.DataFrame:
    """Сопоставляет теги категориям по шаблонам ключевых слов.

    Args:
        tag_names: Очищенные названия тегов.
        patterns: Регулярные выражения категорий.

    Returns:
        Булева таблица (тег × категория), индекс — названия тегов; тег может
        относиться к нескольким категориям или ни к одной.
    """
    names = pd.Series(tag_names, dtype="string").reset_index(drop=True)
    matches = {
        category: names.str.contains(pattern, regex=True).fillna(False).to_numpy(dtype=bool)
        for category, pattern in patterns.items()
    }
    return pd.DataFrame(matches, index=pd.Index(names, name="tag_name"))


def category_tag_counts(
    book_tags: pd.DataFrame, patterns: dict[str, str] = config.BOOK_CATEGORIES
) -> pd.DataFrame:
    """Суммирует счётчики тегов книги по категориям.

    Args:
        book_tags: Очищенные теги `book_id, tag_name, count` (`data.clean_book_tags`).
        patterns: Регулярные выражения категорий.

    Returns:
        Таблица (N_BOOKS × категории без other), индекс — `book_id` 1..N_BOOKS.
    """
    tag_codes, unique_tags = pd.factorize(book_tags["tag_name"])
    matches = tag_category_matrix(pd.Series(unique_tags), patterns)
    book_by_tag = sparse.csr_matrix(
        (
            book_tags["count"].to_numpy(dtype=np.float64),
            (book_tags["book_id"].to_numpy() - 1, tag_codes),
        ),
        shape=(config.N_BOOKS, unique_tags.size),
    )
    counts = book_by_tag @ matches.to_numpy(dtype=np.float64)
    return pd.DataFrame(
        counts,
        index=pd.RangeIndex(1, config.N_BOOKS + 1, name="book_id"),
        columns=list(patterns),
    )


def book_categories(
    book_tags: pd.DataFrame, patterns: dict[str, str] = config.BOOK_CATEGORIES
) -> pd.Series:
    """Определяет тематическую категорию каждой книги.

    Категория — с наибольшей суммой счётчиков тегов (при равенстве — первая по
    порядку `patterns`); книга без тегов из категорий относится к other.
    Категории строятся по метаданным книг, без оценок, поэтому не зависят от split.

    Args:
        book_tags: Очищенные теги `book_id, tag_name, count`.
        patterns: Регулярные выражения категорий.

    Returns:
        Категориальная серия, индекс — `book_id` 1..N_BOOKS.
    """
    counts = category_tag_counts(book_tags, patterns).to_numpy()
    categories = [*patterns, config.CATEGORY_OTHER]
    best = np.where(counts.max(axis=1) > 0, np.argmax(counts, axis=1), len(patterns))
    labels = np.array(categories)[best]
    return pd.Series(
        pd.Categorical(labels, categories=categories),
        index=pd.RangeIndex(1, config.N_BOOKS + 1, name="book_id"),
        name="category",
    )


def _book_stats(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Считает число оценок, сумму и сумму квадратов оценок каждой книги.

    Args:
        train: Обучающие оценки `user_id, book_id, rating`.

    Returns:
        Массивы формы (N_BOOKS,): число оценок, сумма, сумма квадратов.
    """
    index = train["book_id"].to_numpy() - 1
    rating = train["rating"].to_numpy(dtype=np.float64)
    n = np.bincount(index, minlength=config.N_BOOKS).astype(np.float64)
    total = np.bincount(index, weights=rating, minlength=config.N_BOOKS)
    squares = np.bincount(index, weights=rating**2, minlength=config.N_BOOKS)
    return n, total, squares


def _mean_std(n: np.ndarray, total: np.ndarray, squares: np.ndarray) -> tuple[np.ndarray, ...]:
    """Возвращает среднее и стандартное отклонение (ddof=0) по агрегатам.

    Args:
        n: Число значений.
        total: Сумма значений.
        squares: Сумма квадратов значений.

    Returns:
        Среднее и стандартное отклонение; NaN при n = 0.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / n
        variance = np.maximum(squares / n - mean**2, 0.0)
    return mean, np.sqrt(variance)


def book_features(
    train: pd.DataFrame, categories: pd.Series, bayes_m: int = config.BOOK_BAYES_M
) -> pd.DataFrame:
    """Строит признаки книг по обучающей выборке.

    Bayesian average: (Σ r + m·μ) / (n + m), μ — глобальное среднее train;
    для книги без оценок равна μ.

    Args:
        train: Обучающие оценки `user_id, book_id, rating`.
        categories: Категории книг (`book_categories`).
        bayes_m: Сила сглаживания m.

    Returns:
        Таблица с индексом `book_id` 1..N_BOOKS: `n_ratings`, `log_n_ratings`,
        `mean_rating`, `bayes_rating`, `std_rating`, `low_share` (доля оценок 1–2),
        `category`.
    """
    n, total, squares = _book_stats(train)
    mean, std = _mean_std(n, total, squares)
    low = np.bincount(
        train["book_id"].to_numpy() - 1,
        weights=(train["rating"].to_numpy() <= 2).astype(np.float64),
        minlength=config.N_BOOKS,
    )
    global_mean = total.sum() / n.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        low_share = low / n
    return pd.DataFrame(
        {
            "n_ratings": n.astype(np.int64),
            "log_n_ratings": np.log1p(n),
            "mean_rating": mean,
            "bayes_rating": (total + bayes_m * global_mean) / (n + bayes_m),
            "std_rating": std,
            "low_share": low_share,
            "category": categories.reindex(range(1, config.N_BOOKS + 1)).array,
        },
        index=pd.RangeIndex(1, config.N_BOOKS + 1, name="book_id"),
    )


def _user_category_counts(train: pd.DataFrame, categories: pd.Series) -> np.ndarray:
    """Считает число оценок пользователя по категориям книг.

    Args:
        train: Оценки `user_id, book_id, rating`.
        categories: Категории книг (`book_categories`).

    Returns:
        Матрица (N_USERS, число категорий) float64.
    """
    codes = categories.cat.codes.to_numpy()[train["book_id"].to_numpy() - 1]
    n_categories = len(categories.cat.categories)
    flat = (train["user_id"].to_numpy(dtype=np.int64) - 1) * n_categories + codes
    counts = np.bincount(flat, minlength=config.N_USERS * n_categories)
    return counts.reshape(config.N_USERS, n_categories).astype(np.float64)


def user_features(
    train: pd.DataFrame,
    categories: pd.Series,
    threshold: int = config.RELEVANCE_THRESHOLD,
    new_max_ratings: int = config.USER_NEW_MAX_RATINGS,
) -> pd.DataFrame:
    """Строит признаки пользователей по обучающей выборке.

    Мейнстримность — среднее log(1 + n_i) по книгам пользователя, n_i — число
    оценок книги в train. Тип пользователя: new — не более `new_max_ratings`
    оценок (включая отсутствие оценок), иначе active.

    Args:
        train: Обучающие оценки `user_id, book_id, rating`.
        categories: Категории книг (`book_categories`).
        threshold: Порог понравившейся книги.
        new_max_ratings: Порог θ числа оценок нового пользователя.

    Returns:
        Таблица с индексом `user_id` 1..N_USERS: `n_ratings`, `mean_rating`,
        `std_rating`, `liked_share`, `mainstream`, `n_categories`, `user_type`.
    """
    index = train["user_id"].to_numpy() - 1
    rating = train["rating"].to_numpy(dtype=np.float64)
    n = np.bincount(index, minlength=config.N_USERS).astype(np.float64)
    total = np.bincount(index, weights=rating, minlength=config.N_USERS)
    squares = np.bincount(index, weights=rating**2, minlength=config.N_USERS)
    mean, std = _mean_std(n, total, squares)
    liked = np.bincount(index, weights=rating >= threshold, minlength=config.N_USERS)
    book_n, _, _ = _book_stats(train)
    log_popularity = np.log1p(book_n)[train["book_id"].to_numpy() - 1]
    popularity_sum = np.bincount(index, weights=log_popularity, minlength=config.N_USERS)
    with np.errstate(invalid="ignore", divide="ignore"):
        liked_share = liked / n
        mainstream = popularity_sum / n
    n_categories = (_user_category_counts(train, categories) > 0).sum(axis=1)
    user_type = np.where(n <= new_max_ratings, "new", "active")
    return pd.DataFrame(
        {
            "n_ratings": n.astype(np.int64),
            "mean_rating": mean,
            "std_rating": std,
            "liked_share": liked_share,
            "mainstream": mainstream,
            "n_categories": n_categories,
            "user_type": pd.Categorical(user_type, categories=["new", "active"]),
        },
        index=pd.RangeIndex(1, config.N_USERS + 1, name="user_id"),
    )


def category_profile(
    train: pd.DataFrame, categories: pd.Series, threshold: int = config.RELEVANCE_THRESHOLD
) -> pd.DataFrame:
    """Строит профиль пользователя по категориям понравившихся книг.

    Args:
        train: Обучающие оценки `user_id, book_id, rating`.
        categories: Категории книг (`book_categories`).
        threshold: Порог понравившейся книги.

    Returns:
        Таблица (N_USERS × категории) долей понравившихся книг по категориям;
        строка пользователя без понравившихся книг — нули.
    """
    counts = _user_category_counts(train[train["rating"] >= threshold], categories)
    totals = counts.sum(axis=1, keepdims=True)
    shares = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    return pd.DataFrame(
        shares,
        index=pd.RangeIndex(1, config.N_USERS + 1, name="user_id"),
        columns=list(categories.cat.categories),
    )


def category_affinity(
    train: pd.DataFrame,
    categories: pd.Series,
    alpha: float = config.CATEGORY_AFFINITY_ALPHA,
    threshold: int = config.RELEVANCE_THRESHOLD,
) -> np.ndarray:
    """Считает сглаженную affinity пользователей к категориям.

    a(u, c) = (l_uc + α·p_c) / (l_u + α), где l_uc — число понравившихся
    пользователю книг категории c, l_u — всех понравившихся, p_c — доля
    категории среди понравившихся оценок train. При l_u = 0 affinity равна p_c.

    Args:
        train: Обучающие оценки `user_id, book_id, rating`.
        categories: Категории книг (`book_categories`).
        alpha: Сила сглаживания α.
        threshold: Порог понравившейся книги.

    Returns:
        Матрица (N_USERS, число категорий) float32, столбцы в порядке
        `categories.cat.categories`.
    """
    counts = _user_category_counts(train[train["rating"] >= threshold], categories)
    prior = counts.sum(axis=0) / max(counts.sum(), 1.0)
    affinity = (counts + alpha * prior) / (counts.sum(axis=1, keepdims=True) + alpha)
    return affinity.astype(np.float32)


def gather_scores(
    score_fn: Callable[[np.ndarray], np.ndarray],
    user_ids: np.ndarray,
    book_ids: np.ndarray,
    batch_size: int = config.EVAL_BATCH_SIZE,
) -> np.ndarray:
    """Выбирает скоры модели для пар (пользователь, книга).

    Скоры считаются батчами уникальных пользователей по всему каталогу и
    выбираются индексированием, поэтому порядок пар произвольный.

    Args:
        score_fn: Функция скоров батча пользователей формы (n, N_BOOKS),
            например `model.score`.
        user_ids: Идентификаторы пользователей пар.
        book_ids: Идентификаторы книг пар той же длины.
        batch_size: Число уникальных пользователей в батче.

    Returns:
        Скоры пар float32 в исходном порядке.
    """
    user_ids = np.asarray(user_ids)
    book_ids = np.asarray(book_ids)
    unique_users, inverse = np.unique(user_ids, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(
        inverse[order], np.arange(0, unique_users.size + batch_size, batch_size)
    )
    result = np.empty(user_ids.size, dtype=np.float32)
    for batch, start in enumerate(range(0, unique_users.size, batch_size)):
        pairs = order[bounds[batch] : bounds[batch + 1]]
        scores = np.asarray(score_fn(unique_users[start : start + batch_size]))
        result[pairs] = scores[inverse[pairs] - start, book_ids[pairs] - 1]
    return result


def interaction_features(
    user_ids: np.ndarray,
    book_ids: np.ndarray,
    content: ContentBased,
    item_cf: ItemCF,
    funk_svd: FunkSVD,
    affinity: np.ndarray,
    categories: pd.Series,
) -> pd.DataFrame:
    """Строит признаки пар «пользователь × книга».

    Все модели и affinity должны быть обучены на одной выборке (train или
    train_inner).

    Args:
        user_ids: Идентификаторы пользователей пар.
        book_ids: Идентификаторы книг пар той же длины.
        content: Обученная Content-Based модель.
        item_cf: Обученная Item-based CF модель.
        funk_svd: Обученная FunkSVD модель.
        affinity: Affinity пользователей к категориям (`category_affinity`).
        categories: Категории книг (`book_categories`).

    Returns:
        Таблица `user_id`, `book_id`, `content_sim` (cosine к TF-IDF профилю),
        `item_cf_score` (сумма схожестей с понравившимися книгами),
        `category_affinity`, `svd_rating` (прогноз оценки FunkSVD).
    """
    user_ids = np.asarray(user_ids)
    book_ids = np.asarray(book_ids)
    codes = categories.cat.codes.to_numpy()[book_ids - 1]
    return pd.DataFrame(
        {
            "user_id": user_ids,
            "book_id": book_ids,
            "content_sim": gather_scores(content.cosine, user_ids, book_ids),
            "item_cf_score": gather_scores(item_cf.score, user_ids, book_ids),
            "category_affinity": affinity[user_ids - 1, codes],
            "svd_rating": funk_svd.predict(user_ids, book_ids),
        }
    )


def sample_unseen_books(
    user_ids: np.ndarray,
    exclude: sparse.csr_matrix,
    random_state: int = config.RANDOM_STATE,
) -> np.ndarray:
    """Выбирает для каждого пользователя случайную книгу вне заданного множества.

    Книги выбираются равномерно по каталогу; совпавшие с `exclude` перевыбираются.
    Используется для сравнения признаков релевантных и случайных непрочитанных книг.

    Args:
        user_ids: Идентификаторы пользователей (с повторами — по книге на вхождение).
        exclude: Матрица (N_USERS, N_BOOKS) запрещённых книг, например оценки
            train и holdout.
        random_state: Seed генератора.

    Returns:
        Массив `book_id` той же длины, что `user_ids`.
    """
    rng = np.random.default_rng(random_state)
    user_index = np.asarray(user_ids) - 1
    books = rng.integers(0, config.N_BOOKS, size=user_index.size)
    redraw = np.asarray(exclude[user_index, books]).ravel() != 0
    while redraw.any():
        books[redraw] = rng.integers(0, config.N_BOOKS, size=int(redraw.sum()))
        redraw[redraw] = np.asarray(exclude[user_index[redraw], books[redraw]]).ravel() != 0
    return books + 1
