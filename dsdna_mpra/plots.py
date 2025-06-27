import numpy as np
import pandas as pd
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
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
    set_yticklabels: bool = True
) -> AxesImage:

    for (j, i), val in np.ndenumerate(values.to_numpy()):
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
