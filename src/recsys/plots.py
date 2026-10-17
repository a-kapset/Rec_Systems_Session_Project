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
from matplotlib.ticker import PercentFormatter

from recsys import config

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
# Закреплённые цвета моделей: цвет следует за моделью на всех графиках сравнения.
MODEL_COLORS: dict[str, str] = {
    "Popularity (mean)": SERIES[0],
    "Popularity (weighted)": SERIES[1],
    "Content-Based": SERIES[2],
    "Item-based CF": SERIES[3],
    "FunkSVD": SERIES[4],
    "PureSVD": SERIES[5],
    "Hybrid": "#4a3aa7",
}
# Последовательная шкала одного оттенка: head (тёмный) → tail (светлый).
POPULARITY_RAMP = ("#104281", "#2a78d6", "#86b6ef")


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
}


def plot_model_comparison(
    table: pd.DataFrame, columns: tuple[str, ...] = ("precision", "recall", "ndcg", "map")
) -> Figure:
    """Строит сгруппированную столбчатую диаграмму метрик моделей при фиксированном K.

    Левая панель — метрики точности `columns`, правая — HitRate; цвет
    столбца — модель, порядок моделей — порядок строк `table`.

    Args:
        table: Метрики при одном K, индекс — названия моделей.
        columns: Метрики левой панели.

    Returns:
        График.
    """
    fig, (ax_metrics, ax_hit) = plt.subplots(
        1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [3, 1.2]}
    )
    n_models = len(table)
    width = 0.8 / n_models
    x = np.arange(len(columns))
    for i, (model, color) in enumerate(zip(table.index, SERIES, strict=False)):
        offset = (i - (n_models - 1) / 2) * width
        values = table.loc[model, list(columns)].to_numpy(dtype=float)
        ax_metrics.bar(x + offset, values, width * 0.9, color=color, label=model)
    ax_metrics.set_xticks(x, [f"{METRIC_LABELS[c]}@10" for c in columns])
    ax_metrics.set_ylabel("Значение на test")
    ax_metrics.set_title("Метрики точности top-10")
    ax_metrics.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))

    colors = SERIES[:n_models]
    bars = ax_hit.barh(table.index[::-1], table["hit_rate"].to_numpy()[::-1], color=colors[::-1])
    ax_hit.bar_label(bars, fmt="%.3f", padding=3, color=TEXT_SECONDARY, fontsize=9)
    ax_hit.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax_hit.set_xlim(0, table["hit_rate"].max() * 1.25)
    ax_hit.grid(axis="y", visible=False)
    ax_hit.set_xlabel("Доля пользователей с попаданием")
    ax_hit.set_title("HitRate@10")
    fig.tight_layout()
    return fig


def plot_metrics_by_k(
    table: pd.DataFrame, columns: tuple[str, ...] = ("ndcg", "recall", "precision")
) -> Figure:
    """Строит зависимость метрик моделей от длины списка K.

    Args:
        table: Метрики с индексом (model, k).
        columns: Метрики, по одной панели на метрику.

    Returns:
        График.
    """
    fig, axes = plt.subplots(1, len(columns), figsize=(4.3 * len(columns), 4.2), sharex=True)
    models = table.index.get_level_values("model").unique()
    for ax, column in zip(axes, columns, strict=True):
        for model, color in zip(models, SERIES, strict=False):
            part = table.loc[model, column]
            ax.plot(part.index, part.to_numpy(), marker="o", color=color, label=model)
        ax.set_xticks(sorted(table.index.get_level_values("k").unique()))
        ax.set_xlabel("Длина списка K")
        ax.set_ylabel(f"{METRIC_LABELS[column]}@K на test")
        ax.set_title(f"{METRIC_LABELS[column]}@K")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_coverage_vs_ndcg(table: pd.DataFrame) -> Figure:
    """Строит диаграмму рассеяния Coverage@10 и nDCG@10 моделей.

    Args:
        table: Метрики при K = 10, индекс — названия моделей.

    Returns:
        График.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    placed: list[tuple[float, float]] = []
    x_scale, y_scale = table["coverage"].max(), table["ndcg"].max()
    for (model, row), color in zip(table.iterrows(), SERIES, strict=False):
        x, y = row["coverage"], row["ndcg"]
        ax.scatter(x, y, s=90, color=color, edgecolor=SURFACE, linewidth=2)
        # Подпись близкой к уже подписанной точки смещается вниз.
        crowded = any(
            abs(x - px) < 0.05 * x_scale and abs(y - py) < 0.05 * y_scale for px, py in placed
        )
        ax.annotate(
            model,
            (x, y),
            xytext=(8, -14 if crowded else 4),
            textcoords="offset points",
            color=TEXT_PRIMARY,
            fontsize=9,
        )
        placed.append((x, y))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(-0.02, table["coverage"].max() * 1.35)
    ax.set_ylim(0, table["ndcg"].max() * 1.2)
    ax.set_xlabel("Coverage@10 — доля каталога в рекомендациях")
    ax.set_ylabel("nDCG@10 на test")
    ax.set_title("Точность и разнообразие каталога")
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
        for model, color in zip(models, SERIES, strict=False):
            part = table.loc[model]
            ax.errorbar(
                np.arange(len(part)),
                part[column].to_numpy(),
                yerr=part[f"{column}_ci"].to_numpy(),
                marker="o",
                color=color,
                capsize=3,
                label=model,
            )
        ax.set_xticks(np.arange(len(segments)), ticks)
        ax.set_xlabel("Число оценок пользователя")
        ax.set_ylabel(f"{METRIC_LABELS[column]}@10 на test")
        ax.set_title(f"{METRIC_LABELS[column]}@10 по сегментам активности")
    axes[-1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend(fontsize=8, ncols=2)
    fig.tight_layout()
    return fig


def plot_exposure(table: pd.DataFrame, reference: pd.DataFrame) -> Figure:
    """Строит доли позиций top-10 по сегментам популярности книг.

    Верхние строки — доли сегментов в каталоге и в оценках train для сравнения.

    Args:
        table: Результат `evaluation.exposure`, колонки head / mid / tail.
        reference: Опорные распределения с теми же колонками.

    Returns:
        График.
    """
    data = pd.concat([table, reference])[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(data) + 1.6))
    left = np.zeros(len(data))
    for segment, color in zip(data.columns, POPULARITY_RAMP, strict=True):
        values = data[segment].to_numpy(dtype=float)
        ax.barh(
            data.index, values, left=left, color=color, edgecolor=SURFACE, linewidth=2,
            label=segment,
        )  # fmt: skip
        for y, (x0, value) in enumerate(zip(left, values, strict=True)):
            if value >= 0.06:
                text_color = SURFACE if color != POPULARITY_RAMP[-1] else TEXT_PRIMARY
                ax.text(
                    x0 + value / 2, y, f"{value:.0%}", ha="center", va="center",
                    color=text_color, fontsize=9,
                )  # fmt: skip
        left += values
    ax.axhline(len(reference) - 0.5, color=TEXT_SECONDARY, linewidth=1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Доля позиций top-10 (для опорных строк — доля книг или оценок)")
    ax.set_title("Exposure: head / mid / long tail")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
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
            ax.plot(
                x, part[column].to_numpy(), marker="o", color=MODEL_COLORS.get(model), label=model
            )
        ax.set_xticks(x, [str(h) for h in histories])
        ax.set_xlabel("Число оценок пользователя в train")
        ax.set_ylabel(f"{METRIC_LABELS[column]}@10 на test")
        ax.set_title(f"{METRIC_LABELS[column]}@10 при усечённой истории")
    axes[-1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig
