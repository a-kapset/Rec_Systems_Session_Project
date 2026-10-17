# Гибридная система рекомендаций книг (goodbooks-10k)

## Задача

Построение гибридной системы рекомендаций книг на датасете goodbooks-10k: объединение моделей Popularity, Content-Based, Item-based CF и SVD в единый пул кандидатов с весовой схемой по типам пользователей и балансировкой выдачи. Решение расширяет модели и протокол оценки домашнего задания №1 расширенными признаками, генерацией кандидатов и переранжированием.

## Данные

Источник — актуальная версия датасета [goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) на GitHub, загружаемая согласно инструкции на странице [Kaggle](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k). Версия датасета на Kaggle устарела: содержит дубликаты записей и выборку около 100 оценок на книгу вместо полного объёма. Лицензия — CC BY-SA 4.0.

Используются таблицы `ratings.csv` (5 976 479 оценок, `user_id, book_id, rating`, порядок строк хронологический), `books.csv` (10 000 книг), `tags.csv` и `book_tags.csv` (теги книг, присоединяются через `goodreads_book_id`). Описаний книг и явных меток времени в датасете нет.

## Структура репозитория

```
Rec_Systems_Session_Project/
├── README.md                    # компактный отчёт
├── requirements.txt             # закреплённые версии зависимостей
├── pyproject.toml               # пакет recsys в src/, конфигурация ruff и pytest
├── .gitignore
├── scripts/
│   └── download_data.py         # загрузка CSV датасета с GitHub в data/raw
├── src/recsys/
│   ├── config.py                # пути, константы, протокол оценки
│   ├── data.py                  # загрузка, валидация, очистка тегов, матрица взаимодействий
│   ├── split.py                 # per-user temporal split train / test
│   ├── metrics.py               # Precision@K, Recall@K, nDCG@K, HitRate, Coverage, Novelty
│   ├── evaluation.py            # формирование набора оценки, сравнение моделей, сегменты
│   ├── cold_start.py            # усечение истории пользователя, скрытие книг
│   ├── plots.py                 # единый стиль графиков, сохранение в reports/figures
│   └── models/                  # Popularity, Content-Based, Item-based CF, FunkSVD, PureSVD
├── notebooks/
│   └── hybrid_recsys.ipynb      # выполненный ноутбук с результатами по этапам
├── reports/figures/             # графики, используемые в README
└── tests/                       # тесты метрик, split, моделей, оценки, cold start
```

## Воспроизведение

Требования: Python 3.14 (Windows, x86-64); проверено на Python 3.14.3.

```bash
git clone https://github.com/a-kapset/Rec_Systems_Session_Project.git
cd Rec_Systems_Session_Project
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3 (recsys)"
```

Получение данных (файлы сохраняются в `data/raw/`):

- автоматически: `python scripts/download_data.py` — загрузка `ratings.csv`, `books.csv`, `tags.csv`, `book_tags.csv` из репозитория https://github.com/zygmuntz/goodbooks-10k с проверкой числа строк; уже загруженные файлы пропускаются;
- вручную: скачать те же четыре файла из репозитория https://github.com/zygmuntz/goodbooks-10k и поместить в каталог `data/raw/`.

Тесты и выполнение ноутбука (в активированном окружении):

```bash
pytest
jupyter nbconvert --to notebook --execute --inplace notebooks/hybrid_recsys.ipynb
```
