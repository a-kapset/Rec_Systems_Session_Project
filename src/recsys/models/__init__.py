"""Модели рекомендаций с общим интерфейсом `Recommender`.

Popularity, Content-Based, Item-based CF, FunkSVD и PureSVD — базовые модели,
используемые как источники кандидатов и компоненты гибридной системы.
"""

from recsys.models.base import Recommender
from recsys.models.content import ContentBased
from recsys.models.item_cf import ItemCF
from recsys.models.popularity import Popularity
from recsys.models.svd import FunkSVD, PureSVD

__all__ = ["ContentBased", "FunkSVD", "ItemCF", "Popularity", "PureSVD", "Recommender"]
