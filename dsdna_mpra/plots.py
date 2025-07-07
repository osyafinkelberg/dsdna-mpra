import sys
import typing as tp
import numpy as np
import pandas as pd
from numpy.typing import NDArray
import scipy.stats as sps

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
import matplotlib.patches as patches
from matplotlib.patches import Patch
import logomaker

sys.path.insert(0, '..')
from dsdna_mpra import config, motifs  # noqa E402


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
    imshow_args: tp.Optional[dict] = None,
    title_args: tp.Optional[dict] = None,
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
    legend_title: tp.Optional[str] = None,
    normalize_weights: bool = False,
    x_pos_start: float = -1.0,
    x_pos_step: float = 1.0,
    width: float = 0.8,
    legend_keys: tp.Optional[tp.Dict[str, tp.Any]] = None
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


def motif_annotation_plot(
    tf_motifs: tp.Dict[str, np.ndarray],
    tile_id: str,
    tile_contribution_scores: np.ndarray,
    contribution_score_peaks: tp.List[tp.List[int]],
    motif_match_positions: np.ndarray,
    matched_motifs: np.ndarray,
    tf_motif_genes: tp.Dict[str, str],
) -> plt.Figure:
    """
    Create a motif annotation plot with contribution scores, peaks, and motif matches.

    Args:
        tf_motifs (Dict[str, np.ndarray]): Dictionary of motif contribution weight matrices (CWMs).
        tile_id (str): Identifier of the genomic tile.
        tile_contribution_scores (np.ndarray): 200x4 contribution score array.
        contribution_score_peaks (List[List[int]]): List of [start, stop] peak ranges.
        motif_match_positions (np.ndarray): [n_peaks x n_top_matches x 2] array of motif match positions.
        matched_motifs (np.ndarray): [n_peaks x n_top_matches] array of motif names.
        tf_motif_genes (Dict[str, str]): Mapping from motif_id to assigned TF name.

    Returns:
        plt.Figure: Matplotlib figure showing contribution scores and motif annotations.
    """
    num_motif_tracks = matched_motifs.shape[1]
    fig, axes = plt.subplots(
        figsize=(18, 8),
        nrows=1 + num_motif_tracks,
        ncols=1,
        height_ratios=[2] + [1] * num_motif_tracks
    )

    # Panel 1: Contribution Scores (CS)
    scores = tile_contribution_scores.astype(np.float64)
    score_df = pd.DataFrame(scores.T, columns=["A", "C", "G", "T"])
    logomaker.Logo(score_df, ax=axes[0], center_values=False)

    tacs = motifs.rolling_absolute_contribution_scores(scores, window=config.TACS_WINDOW)
    axes[0].plot(np.arange(200), tacs, color='cornflowerblue')
    axes[0].axhline(config.PER_POS_THRESHOLD * config.TACS_WINDOW, linestyle='--', color='red')
    axes[0].set_title(tile_id)
    axes[0].set_ylim([-1, 2.5])
    axes[0].set_xlim([0, 200])
    for start, stop in contribution_score_peaks:
        rect = patches.Rectangle((start, -0.3), stop - start, 0.6,
                                 linewidth=0.5, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect)
    axes[0].tick_params(axis='x', bottom=False, labelbottom=False)
    axes[0].set_ylabel('Contribution Scores')
    axes[0].grid(False)

    # Panels 2+: Motif Matches from CJS ranking
    cs_dfs = motifs.build_contribution_score_dataframes(
        match_positions=motif_match_positions,
        matched_motifs=matched_motifs,
        cwms=tf_motifs,
    )

    for i, cs_df in enumerate(cs_dfs):
        ax = axes[1 + i]
        logomaker.Logo(cs_df, ax=ax, center_values=False)
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        ax.tick_params(axis='y', left=False, labelleft=False)
        ax.set_ylim([-1, 1])
        ax.set_ylabel(f'Top-{i + 1} motif match\n(CJS)')
        ax.grid(False)

        # annotate motif TFs on the plot
        for peak_idx in range(motif_match_positions.shape[0]):
            motif_id = matched_motifs[peak_idx, i]
            tf_label = tf_motif_genes.get(motif_id.removesuffix('_fwd').removesuffix('_rev'), "Unknown TF")
            start, stop = motif_match_positions[peak_idx, i]
            match_pos = int((start + stop) / 2)
            ax.text(
                match_pos, 0.6, tf_label, rotation=0, fontsize=16,
                verticalalignment='bottom', horizontalalignment='center', color='red'
            )

    plt.subplots_adjust(wspace=0, hspace=0.1)
    return fig
