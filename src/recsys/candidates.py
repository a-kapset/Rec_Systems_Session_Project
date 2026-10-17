"""Генерация пула кандидатов из нескольких моделей-источников.

Каждый источник даёт top-N книг пользователя среди непрочитанных: книги из train
исключаются до выбора top-N, поэтому прочитанное не попадает в пул. Списки
источников объединяются без дубликатов, источники книги сохраняются битовой маской.
Пул хранится построчно (смещения `offsets`, как `indptr` у CSR) для векторной
обработки при скоринге и переранжировании.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from recsys import config
from recsys.evaluation import EvalSet
from recsys.models.base import Recommender, top_k_indices

MAX_SOURCES: int = 8


@dataclass(frozen=True)
class CandidatePool:
    """Пул кандидатов для набора пользователей.

    Пары (пользователь, книга) упорядочены по строке пользователя, внутри — по `book_id`.

    Attributes:
        user_ids: Идентификаторы пользователей формы (n_users,).
        offsets: Границы пар пользователей формы (n_users + 1,): пары строки r —
            `offsets[r]:offsets[r + 1]`.
        book_ids: `book_id` кандидатов формы (n_pairs,).
        sources: Битовая маска источников кандидата (бит j — `source_names[j]`).
        source_names: Названия источников.
    """

    user_ids: np.ndarray
    offsets: np.ndarray
    book_ids: np.ndarray
    sources: np.ndarray
    source_names: tuple[str, ...]

    @property
    def sizes(self) -> np.ndarray:
        """Число кандидатов каждого пользователя."""
        return np.diff(self.offsets)

    @property
    def rows(self) -> np.ndarray:
        """Строка пользователя (индекс в `user_ids`) для каждой пары."""
        return np.repeat(np.arange(self.user_ids.size), self.sizes)

    def from_source(self, name: str) -> np.ndarray:
        """Возвращает маску пар, предложенных источником.

        Args:
            name: Название источника.

        Returns:
            Булев массив формы (n_pairs,).
        """
        bit = np.uint8(1 << self.source_names.index(name))
        return (self.sources & bit) > 0

    def mask_label(self, mask: int) -> str:
        """Формирует подпись битовой маски источников.

        Args:
            mask: Битовая маска источников.

        Returns:
            Названия источников через «+».
        """
        return "+".join(name for j, name in enumerate(self.source_names) if int(mask) >> j & 1)

    def source_labels(self, pairs: np.ndarray) -> list[str]:
        """Формирует подписи источников для выбранных пар.

        Args:
            pairs: Индексы пар.

        Returns:
            Названия источников через «+» для каждой пары.
        """
        return [self.mask_label(mask) for mask in self.sources[pairs]]

    def composition(self) -> pd.Series:
        """Считает доли пар пула по сочетаниям источников.

        Returns:
            Доли по убыванию, индекс — сочетание источников через «+».
        """
        shares = pd.Series(self.sources).value_counts(normalize=True)
        shares.index = pd.Index([self.mask_label(mask) for mask in shares.index], name="sources")
        return shares.rename("share")


def select_candidates(
    scores: dict[str, np.ndarray], seen: sparse.csr_matrix, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Выбирает top-N непрочитанных книг каждого источника и объединяет их.

    Кандидат источника — книга со скором строго выше минимального скора
    пользователя у этого источника среди непрочитанных книг: книги с
    минимальным скором (например, с нулевой схожестью Item-CF) не несут сигнала,
    а при постоянном скоре (нет понравившихся книг) источник кандидатов не даёт.

    Args:
        scores: Матрицы скоров источников формы (n_users, N_BOOKS) для одного
            батча пользователей.
        seen: Матрица (n_users, N_BOOKS) прочитанных книг, строки выровнены
            с `scores`.
        n: Число кандидатов от каждого источника.

    Returns:
        Строки пользователей, индексы книг (`book_id - 1`) и битовые маски
        источников; пары без дубликатов, по возрастанию (строка, книга).

    Raises:
        ValueError: Если источников больше `MAX_SOURCES`.
    """
    if len(scores) > MAX_SOURCES:
        raise ValueError(f"не более {MAX_SOURCES} источников кандидатов")
    seen_rows, seen_cols = seen.nonzero()
    rows, items, bits = [], [], []
    for bit, matrix in enumerate(scores.values()):
        values = np.array(matrix, dtype=np.float32)
        values[seen_rows, seen_cols] = -np.inf
        finite = np.isfinite(values)
        row_min = np.where(finite, values, np.inf).min(axis=1)
        top = top_k_indices(values, n)
        top_scores = np.take_along_axis(values, np.maximum(top, 0), axis=1)
        keep = (top >= 0) & (top_scores > row_min[:, None])
        rows.append(np.nonzero(keep)[0])
        items.append(top[keep])
        bits.append(np.full(keep.sum(), 1 << bit, dtype=np.uint8))
    keys = np.concatenate(rows).astype(np.int64) * config.N_BOOKS + np.concatenate(items)
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    sources = np.zeros(unique_keys.size, dtype=np.uint8)
    np.bitwise_or.at(sources, inverse, np.concatenate(bits))
    return unique_keys // config.N_BOOKS, unique_keys % config.N_BOOKS, sources


def build_pool(
    user_ids: np.ndarray,
    rows: np.ndarray,
    items: np.ndarray,
    sources: np.ndarray,
    source_names: tuple[str, ...],
) -> CandidatePool:
    """Собирает пул из пар, упорядоченных по строке пользователя.

    Args:
        user_ids: Идентификаторы пользователей.
        rows: Строки пользователей пар (по неубыванию).
        items: Индексы книг пар (`book_id - 1`).
        sources: Битовые маски источников пар.
        source_names: Названия источников.

    Returns:
        Пул кандидатов.
    """
    offsets = np.searchsorted(rows, np.arange(len(user_ids) + 1))
    return CandidatePool(
        user_ids=np.asarray(user_ids),
        offsets=offsets,
        book_ids=np.asarray(items, dtype=np.int64) + 1,
        sources=np.asarray(sources, dtype=np.uint8),
        source_names=tuple(source_names),
    )


def generate(
    sources: dict[str, Recommender],
    user_ids: np.ndarray,
    seen: sparse.csr_matrix,
    n: int = config.CANDIDATES_PER_SOURCE,
    batch_size: int = config.EVAL_BATCH_SIZE,
) -> CandidatePool:
    """Формирует пул кандидатов из обученных моделей-источников.

    Args:
        sources: Обученные модели по названию источника.
        user_ids: Идентификаторы пользователей.
        seen: Матрица (N_USERS, N_BOOKS) прочитанных книг (оценки train).
        n: Число кандидатов от каждого источника.
        batch_size: Число пользователей в батче.

    Returns:
        Пул кандидатов без прочитанных книг.
    """
    user_ids = np.asarray(user_ids)
    parts = []
    for start in range(0, user_ids.size, batch_size):
        batch = user_ids[start : start + batch_size]
        scores = {name: model.score(batch) for name, model in sources.items()}
        rows, items, bits = select_candidates(scores, seen[batch - 1], n)
        parts.append((rows + start, items, bits))
    rows, items, bits = (np.concatenate(arrays) for arrays in zip(*parts, strict=True))
    return build_pool(user_ids, rows, items, bits, tuple(sources))


def pool_metrics(pool: CandidatePool, eval_set: EvalSet) -> pd.DataFrame:
    """Считает полноту пула и вклад источников на отложенной выборке.

    Args:
        pool: Пул кандидатов, пользователи совпадают с `eval_set.user_ids`.
        eval_set: Набор оценки.

    Returns:
        Таблица по источникам и пулу целиком: `candidates` — среднее число
        кандидатов на пользователя, `recall` — средняя доля релевантных книг,
        попавших в кандидаты (Recall@pool), `unique_share` — доля релевантных
        книг пула, найденных только этим источником (для строки `pool` — только
        одним источником).

    Raises:
        ValueError: Если пользователи пула и набора оценки не совпадают.
    """
    if not np.array_equal(pool.user_ids, eval_set.user_ids):
        raise ValueError("пользователи пула должны совпадать с набором оценки")
    rel_rows, rel_cols = eval_set.relevant.nonzero()
    rows = pool.rows
    keys = rows.astype(np.int64) * config.N_BOOKS + pool.book_ids - 1
    hit = np.isin(keys, rel_rows.astype(np.int64) * config.N_BOOKS + rel_cols)
    n_users, n_relevant = pool.user_ids.size, eval_set.n_relevant
    n_sources = np.bitwise_count(pool.sources)
    single = hit & (n_sources == 1)

    def summary(mask: np.ndarray, unique: np.ndarray) -> dict[str, float]:
        found = np.bincount(rows[mask & hit], minlength=n_users)
        return {
            "candidates": mask.sum() / n_users,
            "recall": float(np.mean(found / n_relevant)),
            "unique_share": (unique & mask).sum() / max(hit.sum(), 1),
        }

    table = {name: summary(pool.from_source(name), single) for name in pool.source_names}
    table["pool"] = summary(np.ones_like(hit), single)
    return pd.DataFrame.from_dict(table, orient="index").rename_axis("source")
