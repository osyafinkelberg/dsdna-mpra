import numpy as np
from numpy.typing import NDArray
from matplotlib.axes import Axes
import scipy.stats as sps


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
