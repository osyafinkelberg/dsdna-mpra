from typing import List
import numpy as np
import pandas as pd
from numpy.typing import NDArray


def binary_separation_stats(
    lower_dist: NDArray[np.float64],
    upper_dist: NDArray[np.float64]
) -> pd.DataFrame:
    """
    Compute binary separation statistics between two distributions using threshold-based analysis.

    Parameters
    ----------
    lower_dist : np.ndarray
        Distribution for the 'negative' class.
    upper_dist : np.ndarray
        Distribution for the 'positive' class.

    Returns
    -------
    pd.DataFrame
        A DataFrame with threshold-based statistics: TN, TP, FP, FN, FPR, TPR, TNR, FDR.
    """
    sep_table = joint_cumulative_sorting(
        [lower_dist, upper_dist],
        ['less', 'greater_equal'],
        fraction=True
    )

    thresholds = sep_table[:, 0]
    true_negative = sep_table[:, 1]
    true_positive = sep_table[:, 2]
    false_positive = 1 - true_negative
    false_negative = 1 - true_positive

    fpr = false_positive / (false_positive + true_negative + 1e-10)
    tpr = true_positive / (true_positive + false_negative + 1e-10)
    tnr = true_negative / (true_negative + false_positive + 1e-10)
    fdr = false_positive / (false_positive + true_positive + 1e-10)

    return pd.DataFrame({
        'threshold': thresholds,
        'TN': true_negative,
        'TP': true_positive,
        'FP': false_positive,
        'FN': false_negative,
        'FPR': fpr,
        'TPR': tpr,
        'TNR': tnr,
        'FDR': fdr
    })


def joint_cumulative_sorting(
    distributions: List[NDArray[np.float64]],
    inequalities: List[str],
    fraction: bool = False
) -> NDArray[np.float64]:
    """
    Perform cumulative counting of multiple distributions over a common set of thresholds.

    Parameters
    ----------
    distributions : list of np.ndarray
        Input arrays representing distributions.
    inequalities : list of str
        Inequality direction for each distribution: one of {'less', 'less_equal', 'greater', 'greater_equal'}.
    fraction : bool, optional
        If True, returns fractions (relative to each distribution's size). Default is False.

    Returns
    -------
    np.ndarray
        2D array where the first column is threshold values,
        and each subsequent column gives cumulative counts or fractions.
    """
    assert all(ineq in {'less', 'less_equal', 'greater', 'greater_equal'} for ineq in inequalities), \
        "Inequality must be one of {'less', 'less_equal', 'greater', 'greater_equal'}"
    assert len(distributions) == len(inequalities), \
        "Number of distributions must match number of inequalities"

    dist_sizes = np.array([dist.size for dist in distributions])
    if dist_sizes.sum() == 0:
        return np.empty((0, len(distributions) + 1))

    indicator_table = np.zeros((dist_sizes.sum(), len(distributions) + 1))
    start = 0
    for i, (dist, size) in enumerate(zip(distributions, dist_sizes)):
        end = start + size
        indicator_table[start:end, 0] = dist
        indicator_table[start:end, i + 1] = 1
        start = end

    # sort by thresholds
    indicator_table = indicator_table[np.argsort(indicator_table[:, 0])]
    indicator_table[:, 1:] = np.cumsum(indicator_table[:, 1:], axis=0)

    unique_mask = np.concatenate([[True], indicator_table[1:, 0] != indicator_table[:-1, 0], [True]])
    cum_table = np.zeros((unique_mask.sum() - 1, len(distributions) + 1))
    cum_table[:, 0] = indicator_table[unique_mask[:-1], 0]

    for i, ineq in enumerate(inequalities):
        col = i + 1
        cum_table[:, col] = indicator_table[unique_mask[1:], col] - np.insert(
            indicator_table[unique_mask[1:], col][:-1], 0, 0
        )
        if ineq == 'less':
            cum_table[:, col] = np.insert(np.cumsum(cum_table[:-1, col]), 0, 0)
        elif ineq == 'less_equal':
            cum_table[:, col] = np.cumsum(cum_table[:, col])
        elif ineq == 'greater':
            cum_table[:, col] = np.insert(np.cumsum(cum_table[:, col][::-1])[::-1][1:], -1, 0)
        else:  # 'greater_equal'
            cum_table[:, col] = np.cumsum(cum_table[:, col][::-1])[::-1]

    if fraction:
        cum_table[:, 1:] /= dist_sizes

    return cum_table
