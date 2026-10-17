"""Визуализация сравнения моделей и эффектов гибридной системы.

Единый стиль графиков (палитра, сетка, подписи), функции построения отдельных
графиков и сохранение в `reports/figures`. Функции принимают заранее
рассчитанные таблицы метрик из `recsys.evaluation` и возвращают
`matplotlib.figure.Figure`.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator, PercentFormatter
from matplotlib.transforms import Bbox

from recsys import config

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
NEUTRAL = "#a3a29d"
# Закреплённые цвета моделей: цвет следует за моделью на всех графиках сравнения.
# Набор моделей одного графика проверен на различимость при нарушениях цветового
# зрения; варианты гибрида имеют общий цвет и различаются типом линии.
HYBRID_PREFIX = "Гибрид"
HYBRID_COLOR = "#4a3aa7"
MODEL_COLORS: dict[str, str] = {
    "Popularity (weighted)": SERIES[0],
    "Content-Based": SERIES[2],
    "Item-based CF": SERIES[3],
    "PureSVD": SERIES[5],
}
# Более длинные названия проверяются первыми.
HYBRID_LINESTYLES: dict[str, str] = {
    "Гибрид + балансировка + long tail": "-.",
    "Гибрид + балансировка": "--",
    "Гибрид без affinity": ":",
    HYBRID_PREFIX: "-",
}


def set_style() -> None:
    """Устанавливает единый стиль графиков matplotlib."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "text.color": TEXT_PRIMARY,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )


def save_figure(fig: Figure, name: str, figures_dir: Path = config.FIGURES_DIR) -> Path:
    """Сохраняет график в PNG.

    Args:
        fig: График.
        name: Имя файла без расширения.
        figures_dir: Каталог для сохранения.

    Returns:
        Путь к сохранённому файлу.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


METRIC_LABELS: dict[str, str] = {
    "precision": "Precision",
    "recall": "Recall",
    "ndcg": "nDCG",
    "map": "MAP",
    "hit_rate": "HitRate",
    "coverage": "Coverage",
    "ild": "ILD",
}
DIVERSITY_LABELS: dict[str, str] = {
    "coverage": "Coverage@10 — доля каталога в рекомендациях",
    "ild": "ILD@10 — разнообразие категорий в списке",
}


def plot_category_shares(table: pd.DataFrame) -> Figure:
    """Строит доли тематических категорий в каталоге и в оценках.

    Args:
        table: Индекс — категории в порядке отображения сверху вниз, колонки
            `books` (доля книг каталога) и `ratings` (доля оценок обучающей выборки).

    Returns:
        График.
    """
    data = table[::-1]
    y = np.arange(len(data))
    height = 0.4
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(data) + 1.4))
    ax.barh(
        y + height / 2, data["books"], height * 0.9, color=SERIES[0], label="Доля книг каталога"
    )
    ax.barh(
        y - height / 2,
        data["ratings"],
        height * 0.9,
        color=SERIES[1],
        label="Доля оценок обучающей выборки",
    )
    ax.set_yticks(y, [name.replace("_", " ") for name in data.index])
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Доля")
    ax.set_title("Тематические категории книг")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def _style(model: str) -> dict[str, object]:
    """Возвращает цвет и тип линии модели: варианты гибрида различаются типом линии."""
    for prefix, linestyle in HYBRID_LINESTYLES.items():
        if model.startswith(prefix):
            return {"color": HYBRID_COLOR, "linestyle": linestyle}
    return {"color": MODEL_COLORS.get(model, NEUTRAL), "linestyle": "-"}


def _place_labels(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, labels: list[str]) -> None:
    """Подписывает точки, выбирая для каждой подписи первое положение без наложений.

    Положения перебираются по порядку (справа сверху, справа снизу, слева сверху,
    слева снизу, справа ниже); наложение проверяется по фактическим размерам
    текста и маркеров на холсте.

    Args:
        ax: Оси с уже заданными пределами.
        xs: Координаты точек по x.
        ys: Координаты точек по y.
        labels: Подписи точек.
    """
    renderer = ax.figure.canvas.get_renderer()
    offsets = (
        (7, 4, "left"),
        (7, -13, "left"),
        (-7, 4, "right"),
        (-7, -13, "right"),
        (7, -24, "left"),
    )
    centers = ax.transData.transform(np.column_stack([xs, ys]))
    occupied = [Bbox.from_bounds(cx - 6, cy - 6, 12, 12) for cx, cy in centers]
    for x, y, label in zip(xs, ys, labels, strict=True):
        for dx, dy, align in offsets:
            text = ax.annotate(
                label, (x, y), xytext=(dx, dy), textcoords="offset points", ha=align,
                color=TEXT_PRIMARY, fontsize=8,
            )  # fmt: skip
            box = text.get_window_extent(renderer)
            if not any(box.overlaps(other) for other in occupied):
                break
            if (dx, dy) != offsets[-1][:2]:
                text.remove()
        occupied.append(box)


def _is_hybrid(model: str) -> bool:
    """Проверяет, относится ли название к варианту гибрида."""
    return model.startswith(HYBRID_PREFIX)


def plot_test_comparison(
    table: pd.DataFrame, columns: tuple[str, ...] = ("ndcg", "recall", "hit_rate", "coverage")
) -> Figure:
    """Строит метрики моделей при K = 10 малыми панелями горизонтальных столбцов.

    Модель обозначена подписью оси, поэтому цвет только выделяет варианты
    гибрида; значения подписаны у столбцов.

    Args:
        table: Метрики при K = 10, индекс — названия моделей в порядке сверху вниз.
        columns: Метрики, по одной панели на метрику.

    Returns:
        График.
    """
    data = table[::-1]
    colors = [HYBRID_COLOR if _is_hybrid(model) else NEUTRAL for model in data.index]
    fig, axes = plt.subplots(
        1, len(columns), figsize=(3.4 * len(columns) + 1.6, 0.42 * len(data) + 1.4), sharey=True
    )
    for ax, column in zip(axes, columns, strict=True):
        values = data[column].to_numpy(dtype=float)
        bars = ax.barh(data.index, values, 0.7, color=colors)
        fmt = "%.4f" if values.max() < 0.2 else "%.3f"
        ax.bar_label(bars, fmt=fmt, padding=3, color=TEXT_SECONDARY, fontsize=8)
        ax.set_xlim(0, values.max() * 1.3)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{METRIC_LABELS[column]}@10")
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    return fig


def plot_tradeoff(table: pd.DataFrame, x_columns: tuple[str, ...] = ("coverage", "ild")) -> Figure:
    """Строит диаграммы рассеяния nDCG@10 против метрик разнообразия.

    Args:
        table: Метрики при K = 10 (`ndcg` и `x_columns`), индекс — названия моделей.
        x_columns: Метрики разнообразия, по одной панели на метрику.

    Returns:
        График.
    """
    fig, axes = plt.subplots(1, len(x_columns), figsize=(6.2 * len(x_columns), 4.6), sharey=True)
    for ax, column in zip(axes, x_columns, strict=True):
        x_scale, y_scale = table[column].max(), table["ndcg"].max()
        ax.set_xlim(0, x_scale * 1.45)
        ax.set_ylim(table["ndcg"].min() * 0.85, y_scale * 1.08)
        for model, row in table.iterrows():
            color = HYBRID_COLOR if _is_hybrid(model) else NEUTRAL
            ax.scatter(
                row[column], row["ndcg"], s=80, color=color, edgecolor=SURFACE, linewidth=2,
                zorder=3,
            )  # fmt: skip
        _place_labels(ax, table[column].to_numpy(), table["ndcg"].to_numpy(), list(table.index))
        ax.set_xlabel(DIVERSITY_LABELS[column])
        ax.set_title(f"nDCG@10 и {METRIC_LABELS[column]}@10")
    axes[0].set_ylabel("nDCG@10 на test")
    fig.tight_layout()
    return fig


def plot_segment_metrics(
    table: pd.DataFrame, columns: tuple[str, ...] = ("ndcg", "hit_rate")
) -> Figure:
    """Строит метрики моделей по сегментам активности пользователей.

    Вертикальные отрезки — 95 % доверительные интервалы.

    Args:
        table: Результат `evaluation.segment_metrics`, индекс (model, segment).
        columns: Метрики, по одной панели на метрику.

    Returns:
        График.
    """
    fig, axes = plt.subplots(1, len(columns), figsize=(6.2 * len(columns), 4.6))
    models = table.index.get_level_values("model").unique()
    segments = table.loc[models[0]]
    ticks = [f"{s}\n({n:,} польз.)".replace(",", " ") for s, n in segments["n_users"].items()]
    for ax, column in zip(axes, columns, strict=True):
        for model in models:
            part = table.loc[model]
            ax.errorbar(
                np.arange(len(part)),
                part[column].to_numpy(),
                yerr=part[f"{column}_ci"].to_numpy(),
                marker="o",
                capsize=3,
                label=model,
                **_style(model),
            )
        ax.set_xticks(np.arange(len(segments)), ticks)
        ax.set_xlabel("Число оценок пользователя")
        ax.set_ylabel(f"{METRIC_LABELS[column]}@10 на test")
        ax.set_title(f"{METRIC_LABELS[column]}@10 по сегментам активности")
    axes[-1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend(fontsize=8, ncols=2)
    fig.tight_layout()
    return fig


def plot_cold_start(table: pd.DataFrame, columns: tuple[str, ...] = ("ndcg", "hit_rate")) -> Figure:
    """Строит метрики моделей в зависимости от длины истории пользователя в train.

    Args:
        table: Метрики с индексом (model, history); history — длина истории, последняя
            точка может быть подписана как полная история.
        columns: Метрики, по одной панели на метрику.

    Returns:
        График.
    """
    fig, axes = plt.subplots(1, len(columns), figsize=(6.2 * len(columns), 4.4))
    models = table.index.get_level_values("model").unique()
    histories = table.index.get_level_values("history").unique()
    x = np.arange(len(histories))
    for ax, column in zip(axes, columns, strict=True):
        for model in models:
            part = table.loc[model].reindex(histories)
            ax.plot(x, part[column].to_numpy(), marker="o", label=model, **_style(model))
        ax.set_xticks(x, [str(h) for h in histories])
        ax.set_xlabel("Число оценок пользователя в train")
        ax.set_ylabel(f"{METRIC_LABELS[column]}@10 на test")
        ax.set_title(f"{METRIC_LABELS[column]}@10 при усечённой истории")
    axes[-1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig
