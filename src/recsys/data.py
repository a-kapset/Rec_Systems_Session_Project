"""Загрузка и валидация данных goodbooks-10k.

Чтение CSV из `data/raw` с компактными типами, сохранение исходного порядка строк
`ratings.csv` (порядок строк соответствует хронологии оценок) и проверка
целостности: диапазоны идентификаторов и оценок, дубликаты, пропуски, ссылочная
целостность между таблицами. Очистка тегов и построение разреженной матрицы
взаимодействий «пользователь × книга».
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from recsys import config


@dataclass(frozen=True)
class GoodbooksData:
    """Набор таблиц датасета goodbooks-10k.

    Attributes:
        ratings: Оценки `user_id, book_id, rating` в исходном (хронологическом) порядке.
        books: Метаданные книг, `book_id` 1..10000.
        tags: Справочник тегов `tag_id, tag_name`.
        book_tags: Связь `goodreads_book_id, tag_id, count`.
    """

    ratings: pd.DataFrame
    books: pd.DataFrame
    tags: pd.DataFrame
    book_tags: pd.DataFrame


def _csv_path(file_name: str, data_dir: Path) -> Path:
    """Возвращает путь к CSV-файлу и проверяет его наличие.

    Args:
        file_name: Имя файла датасета.
        data_dir: Каталог с исходными CSV.

    Returns:
        Путь к файлу.

    Raises:
        FileNotFoundError: Если файл отсутствует.
    """
    path = data_dir / file_name
    if not path.is_file():
        raise FileNotFoundError(
            f"Файл {path} не найден. Загрузка данных: python scripts/download_data.py"
        )
    return path


def load_ratings(data_dir: Path = config.RAW_DATA_DIR) -> pd.DataFrame:
    """Загружает оценки с сохранением исходного порядка строк.

    Args:
        data_dir: Каталог с исходными CSV.

    Returns:
        DataFrame `user_id` (int32), `book_id` (int32), `rating` (int8);
        индекс 0..n-1 соответствует порядку строк в файле.
    """
    return pd.read_csv(
        _csv_path("ratings.csv", data_dir),
        dtype={"user_id": "int32", "book_id": "int32", "rating": "int8"},
    )


def load_books(data_dir: Path = config.RAW_DATA_DIR) -> pd.DataFrame:
    """Загружает метаданные книг.

    Args:
        data_dir: Каталог с исходными CSV.

    Returns:
        DataFrame с метаданными книг, упорядоченный по `book_id`.
    """
    books = pd.read_csv(
        _csv_path("books.csv", data_dir),
        dtype={"book_id": "int32", "goodreads_book_id": "int64", "isbn": "string"},
    )
    return books.sort_values("book_id", ignore_index=True)


def load_tags(data_dir: Path = config.RAW_DATA_DIR) -> pd.DataFrame:
    """Загружает справочник тегов.

    Args:
        data_dir: Каталог с исходными CSV.

    Returns:
        DataFrame `tag_id` (int32), `tag_name` (string).
    """
    return pd.read_csv(
        _csv_path("tags.csv", data_dir),
        dtype={"tag_id": "int32", "tag_name": "string"},
        keep_default_na=False,
    )


def load_book_tags(data_dir: Path = config.RAW_DATA_DIR) -> pd.DataFrame:
    """Загружает связи книг и тегов.

    Args:
        data_dir: Каталог с исходными CSV.

    Returns:
        DataFrame `goodreads_book_id` (int64), `tag_id` (int32), `count` (int32).
    """
    return pd.read_csv(
        _csv_path("book_tags.csv", data_dir),
        dtype={"goodreads_book_id": "int64", "tag_id": "int32", "count": "int32"},
    )


def load_all(data_dir: Path = config.RAW_DATA_DIR) -> GoodbooksData:
    """Загружает все используемые таблицы датасета.

    Args:
        data_dir: Каталог с исходными CSV.

    Returns:
        Набор таблиц `GoodbooksData`.
    """
    return GoodbooksData(
        ratings=load_ratings(data_dir),
        books=load_books(data_dir),
        tags=load_tags(data_dir),
        book_tags=load_book_tags(data_dir),
    )


def validate(data: GoodbooksData) -> dict[str, int]:
    """Проверяет целостность датасета.

    Критические нарушения (пропуски и дубликаты в оценках, выход за диапазоны,
    битые ссылки между таблицами) приводят к исключению. Некритические
    особенности возвращаются в сводке для учёта при очистке.

    Args:
        data: Набор таблиц датасета.

    Returns:
        Сводка: размеры таблиц и число некритических аномалий.

    Raises:
        ValueError: При нарушении целостности данных.
    """
    ratings, books, tags, book_tags = data.ratings, data.books, data.tags, data.book_tags
    errors: list[str] = []

    if ratings.isna().any().any():
        errors.append("пропуски в ratings")
    if ratings.duplicated(["user_id", "book_id"]).any():
        errors.append("дубликаты пар (user_id, book_id) в ratings")
    if not ratings["rating"].between(config.RATING_MIN, config.RATING_MAX).all():
        errors.append("оценки вне диапазона")
    if not ratings["book_id"].between(1, config.N_BOOKS).all():
        errors.append("book_id в ratings вне диапазона")
    if not ratings["user_id"].between(1, config.N_USERS).all():
        errors.append("user_id в ratings вне диапазона")
    if not books["book_id"].is_unique or not books["goodreads_book_id"].is_unique:
        errors.append("неуникальные идентификаторы книг в books")
    if not ratings["book_id"].isin(books["book_id"]).all():
        errors.append("ratings ссылается на отсутствующие книги")
    if not book_tags["goodreads_book_id"].isin(books["goodreads_book_id"]).all():
        errors.append("book_tags ссылается на отсутствующие книги")
    if not book_tags["tag_id"].isin(tags["tag_id"]).all():
        errors.append("book_tags ссылается на отсутствующие теги")
    if errors:
        raise ValueError("Нарушена целостность данных: " + "; ".join(errors))

    return {
        "n_ratings": len(ratings),
        "n_users": int(ratings["user_id"].nunique()),
        "n_books": int(ratings["book_id"].nunique()),
        "n_tags": len(tags),
        "n_book_tags": len(book_tags),
        "book_tags_duplicate_pairs": int(
            book_tags.duplicated(["goodreads_book_id", "tag_id"]).sum()
        ),
        "book_tags_nonpositive_count": int((book_tags["count"] <= 0).sum()),
        "books_missing_original_title": int(books["original_title"].isna().sum()),
    }


def is_service_tag(tag_names: pd.Series) -> pd.Series:
    """Определяет служебные теги, не описывающие содержание книги.

    Служебными считаются теги из стоп-списка, теги по шаблонам (годы, статус
    чтения, личные полки, форматы, личные оценки) и теги без латинских букв.

    Args:
        tag_names: Названия тегов.

    Returns:
        Булева маска той же длины: True для служебных тегов.
    """
    names = tag_names.astype("string").str.lower()
    pattern = "|".join(f"(?:{p})" for p in config.STOP_TAG_PATTERNS)
    return (
        names.isin(config.STOP_TAGS)
        | names.str.contains(pattern, regex=True)
        | ~names.str.contains("[a-z]", regex=True)
    ).fillna(True)


def tag_usage(data: GoodbooksData) -> pd.DataFrame:
    """Сопоставляет теги книгам датасета по `goodreads_book_id`.

    Записи с неположительным `count` отбрасываются.

    Args:
        data: Набор таблиц датасета.

    Returns:
        DataFrame `book_id`, `tag_name`, `count` без очистки названий.
    """
    usage = (
        data.book_tags[data.book_tags["count"] > 0]
        .merge(data.books[["book_id", "goodreads_book_id"]], on="goodreads_book_id")
        .merge(data.tags, on="tag_id")
    )
    return usage[["book_id", "tag_name", "count"]].reset_index(drop=True)


def clean_book_tags(data: GoodbooksData) -> pd.DataFrame:
    """Строит очищенную таблицу тегов книг.

    Удаляются служебные теги, синонимы приводятся к единой форме, повторяющиеся
    пары (книга, тег) объединяются суммированием `count`.

    Args:
        data: Набор таблиц датасета.

    Returns:
        DataFrame `book_id` (int32), `tag_name` (string), `count` (int32),
        упорядоченный по `book_id` и убыванию `count`.
    """
    usage = tag_usage(data)
    usage = usage[~is_service_tag(usage["tag_name"])]
    names = usage["tag_name"].str.lower()
    usage = usage.assign(tag_name=names.replace(config.TAG_SYNONYMS))
    cleaned = usage.groupby(["book_id", "tag_name"], as_index=False, observed=True)["count"].sum()
    cleaned = cleaned.astype({"book_id": "int32", "tag_name": "string", "count": "int32"})
    return cleaned.sort_values(["book_id", "count"], ascending=[True, False], ignore_index=True)


def interaction_matrix(ratings: pd.DataFrame, binary: bool = False) -> sparse.csr_matrix:
    """Строит разреженную матрицу взаимодействий «пользователь × книга».

    Строка `user_id - 1`, столбец `book_id - 1` (идентификаторы непрерывны),
    форма `(N_USERS, N_BOOKS)` независимо от состава выборки.

    Args:
        ratings: Оценки `user_id, book_id, rating`.
        binary: True — значения 1 для всех взаимодействий, иначе — оценки.

    Returns:
        Матрица `csr_matrix` float32.
    """
    values = (
        np.ones(len(ratings), dtype=np.float32)
        if binary
        else ratings["rating"].to_numpy(dtype=np.float32)
    )
    return sparse.csr_matrix(
        (values, (ratings["user_id"].to_numpy() - 1, ratings["book_id"].to_numpy() - 1)),
        shape=(config.N_USERS, config.N_BOOKS),
        dtype=np.float32,
    )
