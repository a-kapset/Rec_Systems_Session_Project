"""Конфигурация проекта: пути, источник данных и глобальные константы.

Единая точка задания путей к данным и отчётам, параметров воспроизводимости
и протокола оценки. Остальные модули импортируют значения отсюда.
"""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
FIGURES_DIR: Path = PROJECT_ROOT / "reports" / "figures"

RANDOM_STATE: int = 42

DATASET_REPO_URL: str = "https://github.com/zygmuntz/goodbooks-10k"
DATASET_RAW_URL: str = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master"

# Ожидаемое число строк (без заголовка) в актуальной версии датасета.
DATASET_FILES: dict[str, int] = {
    "ratings.csv": 5_976_479,
    "books.csv": 10_000,
    "tags.csv": 34_252,
    "book_tags.csv": 999_912,
}

RATING_MIN: int = 1
RATING_MAX: int = 5
N_BOOKS: int = 10_000
N_USERS: int = 53_424

RELEVANCE_THRESHOLD: int = 4
TEST_FRACTION: float = 0.2
MIN_RATINGS_TO_SPLIT: int = 5
K_VALUES: tuple[int, ...] = (5, 10, 20)
EVAL_N_USERS: int = 10_000
EVAL_BATCH_SIZE: int = 2_000

# Гиперпараметры (выбраны на valid при обучении на train_inner). Порог числа оценок
# Popularity масштабируется пропорционально размеру обучающей выборки.
POPULARITY_MIN_RATINGS: int = 7_000
# 7 000 · |train| / |train_inner| = 7 000 · 4 781 208 / 3 824 978.
POPULARITY_MIN_RATINGS_TRAIN: int = 8_750
POPULARITY_WR_M: int = 100_000
CONTENT_N_TAGS: int | None = None
ITEM_CF_SIMILARITY: str = "adjusted_cosine"
ITEM_CF_NEIGHBORS: int = 50
ITEM_CF_SHRINKAGE: float = 50.0
SVD_N_FACTORS: int = 100
SVD_N_EPOCHS: int = 30
SVD_LR_ALL: float = 0.005
SVD_REG_ALL: float = 0.05
SVD_MIN_ITEM_RATINGS: int = 1
PURE_SVD_N_FACTORS: int = 200

# Симуляция cold start: число пользователей с усечённой историей, длины истории,
# доля каталога, скрываемая как новые книги.
COLD_N_USERS: int = 5_000
COLD_HISTORY_LENGTHS: tuple[int, ...] = (1, 3, 5, 10, 20)
COLD_ITEM_SHARE: float = 0.05

# Сегменты пользователей по числу оценок: границы [left, right).
USER_ACTIVITY_BINS: tuple[int, ...] = (0, 50, 100, 150, 10_000)
USER_ACTIVITY_LABELS: tuple[str, ...] = ("<50", "50–99", "100–149", "≥150")

# Сегменты книг по популярности: head — книги, дающие первые 50 % оценок,
# tail — книги за пределами первых 80 % оценок, mid — остальные.
HEAD_RATING_SHARE: float = 0.5
TAIL_RATING_SHARE: float = 0.8

# Признаки. Тип пользователя «новый» — не более USER_NEW_MAX_RATINGS оценок в train;
# сила сглаживания средней оценки книги к глобальному среднему (Bayesian average);
# сила сглаживания affinity пользователя к категории к доле категории в train.
USER_NEW_MAX_RATINGS: int = 10
BOOK_BAYES_M: int = 50
CATEGORY_AFFINITY_ALPHA: float = 10.0

# Генерация кандидатов: источники пула и число кандидатов от каждого источника.
CANDIDATE_SOURCES: tuple[str, ...] = ("popularity", "content", "item_cf", "pure_svd")
CANDIDATES_PER_SOURCE: int = 50

# Компоненты взвешенного гибрида и стартовые веса по типам пользователей
# (уточняются подбором на valid). У новых пользователей больше вес Popularity.
HYBRID_COMPONENTS: tuple[str, ...] = (
    "popularity", "content", "item_cf", "pure_svd", "funk_svd", "affinity",
)  # fmt: skip
HYBRID_WEIGHTS: dict[str, dict[str, float]] = {
    "active": {
        "popularity": 0.05, "content": 0.15, "item_cf": 0.4,
        "pure_svd": 0.2, "funk_svd": 0.1, "affinity": 0.1,
    },
    "new": {
        "popularity": 0.4, "content": 0.2, "item_cf": 0.15,
        "pure_svd": 0.05, "funk_svd": 0.05, "affinity": 0.15,
    },
}  # fmt: skip

# Балансировка top-K: head — доля самых популярных книг каталога по числу оценок
# в train; квоты — не более q книг одной категории и не менее t книг вне head.
BALANCE_HEAD_SHARE: float = 0.1
BALANCE_MAX_PER_CATEGORY: int = 3
BALANCE_MIN_TAIL: int = 3

# Тематические категории книг: шаблоны ключевых слов по очищенным тегам. Тег
# относится ко всем категориям, шаблоны которых он содержит; категория книги —
# с наибольшей суммой счётчиков таких тегов, при отсутствии совпадений — other.
CATEGORY_OTHER: str = "other"
BOOK_CATEGORIES: dict[str, str] = {
    "fantasy": (
        r"fantasy|magic|dragons?$|^fae$|faer(?:y|ies)|fairies|witch|wizard|vampire|vamps"
        r"|werewol|shifters?$|shapeshift|paranormal|supernatural|^angels?$|demons?$"
        r"|mytholog|fairy-tales?|harry-potter|discworld|pratchett"
    ),
    "science_fiction": (
        r"sci-?fi|science-fiction|^sff?$|^sf-|dystop|apocalyp|^space|space-opera|aliens?$"
        r"|time-travel|steampunk|cyberpunk|futuristic|star-wars|star-trek|speculative"
    ),
    "mystery_thriller": (
        r"myster|thriller|crime|detective|suspense|murder|espionage|^spy|spies|noir"
        r"|agatha-christie|james-patterson|grisham"
    ),
    "romance": (
        r"romance|romantic|chick-?lit|chic-lit|^love$|love-stor|erotic|bdsm|alpha-male"
        r"|new-adult|book-boyfriend|nicholas-sparks|nora-roberts|kristen-ashley|steamy|smut"
    ),
    "young_adult": r"young-adult|^ya(?:-|$)|teen|high-school",
    "children": (
        r"(?<!lee-)child|kids|juvenile|picture-books?|middle-grade|chapter-books|newbery"
        r"|caldecott|seuss|roald-dahl|bedtime"
    ),
    "classics": (
        r"classic|clàssics|1001|rory-gilmore|shakespeare|victorian|19th-century|pulitzer"
        r"|banned-books|russian-lit|english-lit|british-lit|american-lit"
    ),
    "literary_fiction": (
        r"^contemporary$|contemporary-fiction|literary|^literature$|realistic-fiction"
        r"|general-fiction|women-s-fiction|womens-fiction|book-group|modern-fiction"
    ),
    "historical_fiction": (
        r"historical|^ww|world-war|holocaust|tudors?$|medieval|regency|civil-war|^war$"
    ),
    "horror": r"horror|ghosts?$|zombies?$|stephen-king|^king$|koontz|gothic|scary|creepy|haunted",
    "non_fiction": (
        r"non-fiction|(?<!alternate-)history|science(?!-fiction)|psycholog|politic|economics"
        r"|sociology|(?<!time-)travel|essays|true-crime|^food|cooking|cookbooks?|health"
        r"|journalism|reference|nature|math|physics|biology|neuroscience|medicine"
        r"|education|feminism|environment|technology|parenting|^art$|music"
    ),
    "biography_memoir": r"biograph|memoir",
    "self_help_business": (
        r"self-help|self-improvement|personal-development|business|leadership|productivity"
        r"|management|finance|money|entrepreneur|marketing|motivation|career|investing"
    ),
    "poetry_drama": r"poetry|poems?$|^plays?$|theatre|theater",
    "comics": (
        r"comic|cómic|graphic-novel|manga|mangá|^graphic$|marvel|dc-comics|superhero|batman"
    ),
    "religion_philosophy": (
        r"religio|christian|theology|spiritual|philosoph|faith|bible|buddhis|islam|jewish"
        r"|judaism|(?:^|-)god(?:-|$)|church|inspirational"
    ),
}

# Служебные теги (статус чтения, владение, формат, личные оценки и списки).
STOP_TAGS: frozenset[str] = frozenset(
    {
        "abandoned", "all-time-favorites", "audible", "audio", "bookclub", "book-club",
        "books", "bookshelf", "borrowed", "calibre", "currently-reading", "default",
        "did-not-finish", "didn-t-finish", "dnf", "english", "favorite", "favorites",
        "favourite", "favourites", "finished", "have", "i-own", "library", "maybe",
        "must-read", "nook", "on-hold", "own", "owned", "own-it", "paperback",
        "hardcover", "re-read", "reread", "recommended", "reviewed", "series",
        "tbr", "unfinished", "unread", "wish-list", "wishlist", "part-of-a-series",
    }
)  # fmt: skip

# Шаблоны служебных тегов: годы, личные полки, форматы, статусы, звёзды.
STOP_TAG_PATTERNS: tuple[str, ...] = (
    r"(?:19|20)\d\d",
    r"^(?:to|want|need|wanna)-",
    r"(?:^|-)(?:my|i|own|owned|have|mine)(?:-|$)",
    r"read(?:ing)?-(?:in|20|19)|(?:^|-)read-\d|^read$|-reads?$|^reads?-",
    r"fav",
    r"(?:^|-)(?:kindle|e-?books?|audio-?books?|audio|audible|nook|pdf|epub|ibooks?)(?:-|$)",
    r"(?:^|-)(?:library|shelf|shelves|bookshelf|home|wishlist|wish-list)(?:-|$)",
    r"\d-stars?|^stars?$",
    r"shelfari|goodreads|calibre|default|book-?club|dnf|not-finish|unfinish",
)

# Нормализация синонимичных тегов.
TAG_SYNONYMS: dict[str, str] = {
    "science-fiction": "sci-fi",
    "scifi": "sci-fi",
    "ya": "young-adult",
    "ya-fiction": "young-adult",
    "young-adult-fiction": "young-adult",
    "nonfiction": "non-fiction",
    "classic": "classics",
    "classic-literature": "classics",
    "mysteries": "mystery",
    "thrillers": "thriller",
    "novel": "novels",
    "humour": "humor",
    "childrens": "children",
    "children-s": "children",
    "kids": "children",
    "childrens-books": "children",
    "graphic-novel": "graphic-novels",
    "memoirs": "memoir",
    "dystopia": "dystopian",
    "vampire": "vampires",
    "children-s-books": "children",
    "biographies": "biography",
}
