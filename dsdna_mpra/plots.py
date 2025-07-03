import typing as tp
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.patches import Patch
import scipy.stats as sps
from typing import Optional


def violin(
    ax: Axes,
    values: NDArray[np.float64],
    pos: float,
    violincolor: str = 'royalblue',
    bw_method: float = 0.1,
    width_factor: float = 25,
    quantile_cutoff: float = 0.01,
    box_width: float = 0.15,
    text: bool = True
) -> None:
    values = np.asarray(values)
    values = values[~np.isnan(values)]  # Remove NaNs

    if len(values) == 0:
        return

    if len(values) > 10:
        kde = sps.gaussian_kde(values, bw_method=bw_method)
        xx = np.linspace(
            np.quantile(values, quantile_cutoff),
            np.quantile(values, 1 - quantile_cutoff),
            1000
        )
        k = kde(xx) * width_factor
        ax.plot(k + pos, xx, color='black', linewidth=1.5)
        ax.plot(-k + pos, xx, color='black', linewidth=1.5)
        ax.fill_betweenx(xx, -k + pos, k + pos, color=violincolor, alpha=0.6)
    else:
        x = np.random.normal(pos, 0.02, size=len(values))
        if len(values) == 1:
            x = [pos]
        ax.scatter(x, values, color=violincolor, alpha=0.7, s=70)

    box(ax, values, pos, width=box_width, text=text)


def box(
    ax: Axes,
    values: NDArray[np.float64],
    pos: float,
    width: float,
    border_color: str = 'black',
    text: bool = True
) -> None:
    values = np.asarray(values)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return

    boxprops = dict(color=border_color, facecolor='white', linewidth=1.5)
    if len(values) <= 10:
        boxprops['facecolor'] = (0, 0, 0, 0)

    ax.boxplot(
        values,
        positions=[pos],
        notch=True,
        showfliers=False,
        whis=(5, 95),
        showcaps=False,
        patch_artist=True,
        widths=width,
        medianprops=dict(color=border_color, linewidth=3),
        boxprops=boxprops,
        whiskerprops=dict(color=border_color, linewidth=1.5)
    )

    if text:
        ax.text(x=pos, y=np.quantile(values, 0.55), s=str(len(values)),
                ha='center', fontsize=10)


def heatmap_with_stats(
    ax: Axes,
    values: pd.DataFrame,
    imshow_args: Optional[dict] = None,
    title_args: Optional[dict] = None,
    text_fontsize: int = 15,
    set_xticklabels: bool = True,
    set_yticklabels: bool = True,
    text_values: np.ndarray | None = None,
) -> AxesImage:

    if text_values is None:
        text_values = values.to_numpy()
    else:
        assert text_values.shape == values.to_numpy().shape

    for (j, i), val in np.ndenumerate(text_values):
        if pd.isna(val):
            continue
        if np.isclose(val, int(val)):
            text = f"{int(val)}"
        else:
            text = f"{val:.2f}"
        ax.text(
            i, j, text,
            ha='center', va='center',
            fontsize=text_fontsize, color='black'
        )

    if imshow_args is None:
        imshow_args = {'cmap': 'viridis', 'vmin': 0}

    img = ax.imshow(values, interpolation='nearest', aspect='auto', **imshow_args)

    if title_args is not None:
        ax.set_title(**title_args)

    if set_xticklabels:
        ax.set_xticks(np.arange(values.shape[1]))
        ax.set_xticklabels(values.columns, fontsize=10)
        ax.xaxis.tick_top()
    if set_yticklabels:
        ax.set_yticks(np.arange(values.shape[0]))
        ax.set_yticklabels(values.index, fontsize=15)

    ax.grid(False)
    return img


def stacked_bar_plot(
    ax: Axes,
    dataframe: pd.DataFrame,
    x_value: str,
    hue: str,
    weight: str,
    color: str,
    legend_title: Optional[str] = None,
    normalize_weights: bool = False,
    x_pos_start: float = -1.0,
    x_pos_step: float = 1.0,
    width: float = 0.8,
    legend_keys: Optional[tp.Dict[str, tp.Any]] = None
) -> None:
    if legend_keys is None:
        legend_keys = {}

    legend_entries = set()
    x_labels = []
    x_pos = x_pos_start

    for x_val, group in dataframe.groupby(x_value, sort=False):
        x_labels.append(x_val)
        x_pos += x_pos_step
        y_base = 0.0

        hues = group[hue].values
        weights = group[weight].values
        bar_colors = (
            np.full(group.shape[0], color) if color not in group.columns else group[color].values
        )

        if normalize_weights:
            total_weight = weights.sum()
            weights = weights / total_weight if total_weight != 0 else weights

        for i in range(len(group)):
            ax.bar(x_pos, weights[i], width=width, bottom=y_base, color=bar_colors[i])
            legend_entries.add((bar_colors[i], hues[i]))
            y_base += weights[i]

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=90, fontsize=15)

    if legend_title:
        sorted_legend = sorted(legend_entries, key=lambda x: str(x[1]))
        legend_handles = [
            Patch(facecolor=color, edgecolor=color, label=label) for color, label in sorted_legend
        ]
        ax.legend(handles=legend_handles, title=legend_title, loc='upper left', **legend_keys)
