"""Загрузка CSV-файлов датасета goodbooks-10k с GitHub в `data/raw`.

Источник — актуальная версия датасета в репозитории zygmuntz/goodbooks-10k.
Уже скачанные файлы с корректным числом строк пропускаются; после загрузки
число строк сверяется с ожидаемым.
"""

import argparse
import logging
import shutil
import urllib.request
from pathlib import Path

from recsys import config

logger = logging.getLogger(__name__)


def count_rows(path: Path) -> int:
    """Считает число строк данных в CSV без учёта заголовка.

    Args:
        path: Путь к CSV-файлу.

    Returns:
        Число строк данных.
    """
    with path.open("rb") as file:
        return sum(1 for _ in file) - 1


def download_file(file_name: str, target_dir: Path, force: bool = False) -> Path:
    """Скачивает один файл датасета и проверяет число строк.

    Args:
        file_name: Имя файла в репозитории датасета.
        target_dir: Каталог назначения.
        force: Повторная загрузка при наличии файла.

    Returns:
        Путь к скачанному файлу.

    Raises:
        ValueError: Если число строк не совпадает с ожидаемым.
    """
    expected = config.DATASET_FILES[file_name]
    path = target_dir / file_name
    if path.is_file() and not force and count_rows(path) == expected:
        logger.info("%s: уже загружен, пропуск", file_name)
        return path

    url = f"{config.DATASET_RAW_URL}/{file_name}"
    tmp_path = path.with_suffix(".part")
    logger.info("%s: загрузка %s", file_name, url)
    with urllib.request.urlopen(url, timeout=60) as response, tmp_path.open("wb") as out:
        shutil.copyfileobj(response, out)

    n_rows = count_rows(tmp_path)
    if n_rows != expected:
        tmp_path.unlink()
        raise ValueError(f"{file_name}: {n_rows} строк вместо ожидаемых {expected}")
    tmp_path.replace(path)
    logger.info("%s: %d строк", file_name, n_rows)
    return path


def main() -> None:
    """Точка входа: загрузка всех используемых файлов датасета."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="перезагрузить существующие файлы")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for file_name in config.DATASET_FILES:
        download_file(file_name, config.RAW_DATA_DIR, force=args.force)


if __name__ == "__main__":
    main()
