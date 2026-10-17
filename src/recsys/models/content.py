"""Контентная модель: TF-IDF профили книг по названию и тегам.

Текстовый профиль книги — `original_title` (при отсутствии — `title`) и top-N
очищенных тегов по числу присвоений; тег записывается одним токеном
(`sci-fi` → `sci_fi`). Профили векторизуются `TfidfVectorizer` с L2-нормировкой,
поэтому cosine similarity равна скалярному произведению разреженных векторов.
Персонализированный вариант ранжирует книги по близости к профилю пользователя —
сумме TF-IDF векторов книг, оценённых в train не ниже порога, с весом-оценкой.
"""

from typing import Self

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from recsys import config
from recsys.data import interaction_matrix
from recsys.models.base import Recommender, top_k_indices


def book_titles(books: pd.DataFrame) -> pd.Series:
    """Возвращает название книги для профиля.

    Args:
        books: Метаданные книг.

    Returns:
        `original_title`, при пустом значении — `title`; индекс — `book_id`.
    """
    original = books["original_title"].astype("string").str.strip().replace("", pd.NA)
    return original.fillna(books["title"].astype("string")).set_axis(books["book_id"])


def build_profiles(
    books: pd.DataFrame, book_tags: pd.DataFrame, n_tags: int | None = config.CONTENT_N_TAGS
) -> pd.Series:
    """Строит текстовые профили книг из названия и тегов.

    Args:
        books: Метаданные книг (`book_id`, `original_title`, `title`).
        book_tags: Очищенные теги `book_id, tag_name, count` (`data.clean_book_tags`),
            упорядоченные по убыванию `count` внутри книги.
        n_tags: Число тегов с наибольшим `count` на книгу; None — все теги.

    Returns:
        Профили, индекс — `book_id` в порядке `books`.
    """
    tags = book_tags
    if n_tags is not None:
        tags = tags[tags.groupby("book_id").cumcount() < n_tags]
    tokens = tags["tag_name"].str.replace(r"\W+", "_", regex=True).str.strip("_")
    tag_text = tokens.groupby(tags["book_id"]).agg(" ".join)
    titles = book_titles(books)
    return (titles + " " + tag_text.reindex(titles.index).fillna("")).str.strip()


class ContentBased(Recommender):
    """Content-Based модель на TF-IDF профилях книг.

    Attributes:
        n_tags: Число тегов в профиле книги.
        vectorizer: Обученный `TfidfVectorizer`.
        item_vectors: L2-нормированная матрица (N_BOOKS, n_terms), строка — `book_id - 1`.
    """

    def __init__(
        self,
        books: pd.DataFrame,
        book_tags: pd.DataFrame,
        n_tags: int | None = config.CONTENT_N_TAGS,
    ) -> None:
        """Строит TF-IDF векторы книг.

        Args:
            books: Метаданные книг, упорядоченные по `book_id`.
            book_tags: Очищенные теги книг (`data.clean_book_tags`).
            n_tags: Число тегов с наибольшим `count` на книгу; None — все теги.
        """
        self.n_tags = n_tags
        self.name = f"Content-Based[tags={n_tags or 'all'}]"
        self.books = books[["book_id", "title", "authors"]].reset_index(drop=True)
        self.profiles = build_profiles(books, book_tags, n_tags)
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            token_pattern=r"(?u)\b\w\w+\b",
            min_df=2,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.item_vectors: sparse.csr_matrix = self.vectorizer.fit_transform(self.profiles)
        self._user_weights: sparse.csr_matrix | None = None

    def fit(self, train: pd.DataFrame) -> Self:
        """Запоминает веса книг для профилей пользователей.

        Вес книги в профиле — оценка, если она не ниже порога релевантности, иначе 0.

        Args:
            train: Обучающие оценки `user_id, book_id, rating`.

        Returns:
            Обученная модель.
        """
        liked = train[train["rating"] >= config.RELEVANCE_THRESHOLD]
        self._user_weights = interaction_matrix(liked)
        return self

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        """Считает cosine-близость книг к профилям пользователей (с точностью до нормы профиля).

        Args:
            user_ids: Идентификаторы пользователей.

        Returns:
            Матрица (len(user_ids), N_BOOKS).
        """
        profiles = self._user_weights[np.asarray(user_ids) - 1] @ self.item_vectors
        return (profiles @ self.item_vectors.T).toarray()

    def get_similar_books(self, book_id: int, n: int = 5) -> pd.DataFrame:
        """Находит N книг, наиболее близких к заданной по cosine similarity.

        Args:
            book_id: Идентификатор книги.
            n: Число похожих книг.

        Returns:
            DataFrame: `book_id`, `title`, `authors`, `similarity` по убыванию близости.
        """
        index = book_id - 1
        similarity = linear_kernel(self.item_vectors[index], self.item_vectors)
        similarity[0, index] = -np.inf
        top = top_k_indices(similarity, n)[0]
        result = self.books.iloc[top].reset_index(drop=True)
        return result.assign(similarity=similarity[0, top])
