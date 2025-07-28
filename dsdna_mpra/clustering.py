import typing as tp
import numpy as np
import pandas as pd
from numpy.typing import NDArray


def merge(positions: NDArray[np.int64]) -> NDArray[np.int64]:
    """
    Merge overlapping or adjacent intervals.

    Parameters:
        positions (np.ndarray): A 2D NumPy array of shape (n, 2), where each row is [start, end].
                                Intervals may be unsorted and overlapping.

    Returns:
        np.ndarray: A 2D NumPy array of merged intervals.
    """
    if positions.size == 0:
        return positions
    starts = positions[:, 0]
    ends = positions[:, 1]
    borders = np.concatenate([
        np.stack([starts, np.ones_like(starts)], axis=1),
        np.stack([ends, np.zeros_like(ends)], axis=1)
    ])
    borders = borders[np.argsort(borders[:, 0], kind='stable')]
    merged = []
    depth = 0
    current_start = None
    for position, is_start in borders:
        if is_start:
            if depth == 0:
                current_start = position
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                merged.append([current_start, position])

    return np.array(merged, dtype=positions.dtype)


def vectorized_pearsonr(
    lhs_matrix: NDArray[np.float64],
    rhs_matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Computes the Pearson correlation coefficient between each pair of columns in lhs_matrix and rhs_matrix,
    using a fully vectorized approach, while ignoring NaNs and Infs.

    Parameters
    ----------
    lhs_matrix : np.ndarray of shape (N, D1)
        Left-hand side input matrix where columns are variables and rows are observations.
    rhs_matrix : np.ndarray of shape (N, D2)
        Right-hand side input matrix with the same number of rows as lhs_matrix.

    Returns
    -------
    corr_matrix : np.ndarray of shape (D2, D1)
        Matrix of Pearson correlation coefficients between each column of rhs_matrix and lhs_matrix.
        Entry (i, j) is the correlation between rhs_matrix[:, i] and lhs_matrix[:, j].
    """
    lhs_valid = np.isfinite(lhs_matrix)  # shape (N, D1)
    rhs_valid = np.isfinite(rhs_matrix)  # shape (N, D2)
    valid_mask = lhs_valid[:, :, None] & rhs_valid[:, None, :]  # shape (N, D1, D2)
    lhs_matrix_clean = np.where(lhs_valid, lhs_matrix, 0.0)
    rhs_matrix_clean = np.where(rhs_valid, rhs_matrix, 0.0)
    lhs = lhs_matrix_clean[:, :, None]  # (N, D1, 1)
    rhs = rhs_matrix_clean[:, None, :]  # (N, 1, D2)
    valid_counts = np.sum(valid_mask, axis=0)  # (D1, D2)
    lhs_sum = np.sum(lhs * valid_mask, axis=0)
    rhs_sum = np.sum(rhs * valid_mask, axis=0)
    dot_product = np.sum(lhs * rhs * valid_mask, axis=0)
    lhs_sq_sum = np.sum(lhs**2 * valid_mask, axis=0)
    rhs_sq_sum = np.sum(rhs**2 * valid_mask, axis=0)
    lhs_var = valid_counts * lhs_sq_sum - lhs_sum**2
    rhs_var = valid_counts * rhs_sq_sum - rhs_sum**2
    denominator = np.sqrt(lhs_var * rhs_var)
    numerator = valid_counts * dot_product - lhs_sum * rhs_sum
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_matrix = numerator / denominator
    return corr_matrix.T  # shape (D2, D1)


def find_sorted_label_boundaries(labels: NDArray[np.integer]) -> NDArray[np.int64]:
    """
    Given a 1D array of labels, returns the indices where label changes occur
    after sorting. These are the boundaries between label groups.

    Parameters
    ----------
    labels : np.ndarray of shape (N,)
        Array of integer or categorical labels.

    Returns
    -------
    borders : np.ndarray of shape (K,)
        Indices in the sorted array where the label changes.
    """
    sorted_labels = np.sort(labels)
    changes = sorted_labels[1:] != sorted_labels[:-1]
    return np.nonzero(changes)[0] + 1


def supervised_binary_clustering(
    tile_lfc_array: NDArray[np.float64],
    coverage_threshold: float = 0.8,
    binarization_threshold: float = 0.5
) -> tp.Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """
    Clusters CRE tiles into binary activity profiles based on thresholded rank scores,
    then selects the largest classes covering a specified portion of the dataset.

    Parameters
    ----------
    tile_lfc_array : np.ndarray of shape (N, C)
        Log fold-change activity values per tile (N tiles × C conditions).
    coverage_threshold : float, default=0.8
        Fraction of total tiles to cover using the largest binary activity classes.
    binarization_threshold : float, default=0.5
        Threshold for binarizing the normalized ranks.

    Returns
    -------
    matrix : np.ndarray
        Ranked and filtered activity matrix (tiles in selected large classes).
    borders : np.ndarray
        Indices where class boundaries occur within the returned matrix.
    selected_class_ids : np.ndarray
        Binary class identifiers retained after filtering.
    """

    # rank-normalize tile activities
    tile_activity_ranks = tile_lfc_array.argsort(axis=0).argsort(axis=0) / tile_lfc_array.shape[0]

    # binarize ranks and encode as integer class IDs
    binary_matrix = (tile_activity_ranks >= binarization_threshold).astype(int)
    binary_class_ids = np.array([
        int("".join(bits.astype(str)), 2) for bits in binary_matrix
    ])

    # prepare DataFrame to track class info
    tile_activities = pd.DataFrame({'rank_class': binary_class_ids})
    class_counts = tile_activities['rank_class'].value_counts().to_dict()
    class_to_size = np.vectorize(class_counts.get)
    tile_activities['rank_class_size'] = class_to_size(tile_activities['rank_class'])

    # sort and re-categorize rank classes by decreasing size
    sorted_classes = (
        tile_activities
        .sort_values('rank_class_size', ascending=False)['rank_class']
        .drop_duplicates()
    )
    tile_activities['rank_class'] = (
        tile_activities['rank_class']
        .astype('category')
        .cat.set_categories(sorted_classes, ordered=True)
    )

    borders = np.concatenate([
        [0],
        find_sorted_label_boundaries(binary_class_ids),
        [binary_class_ids.size]
    ])
    class_sizes = np.diff(borders)

    # identify largest classes covering at least `coverage_threshold` fraction of data
    sorted_size_indices = class_sizes.argsort()[::-1]
    cumulative_sizes = class_sizes[sorted_size_indices].cumsum()
    cutoff_index = np.searchsorted(cumulative_sizes, coverage_threshold * class_sizes.sum())

    retain_mask = np.zeros_like(class_sizes, dtype=bool)
    retain_mask[sorted_size_indices[:cutoff_index + 1]] = True
    selected_class_ids = np.unique(binary_class_ids)[retain_mask]

    # select tiles in retained classes and sort within each class
    is_large_class = np.isin(binary_class_ids, selected_class_ids)
    size_key = (
        tile_activities[is_large_class]
        .reset_index(drop=True)
        .sort_values('rank_class', ascending=True)
        .index
        .values
    )

    matrix = tile_activity_ranks[is_large_class][size_key]

    # compute class borders within the reduced matrix
    final_borders = np.cumsum(
        sorted(class_to_size(tile_activities[is_large_class]['rank_class'].unique()), reverse=True)
    ) - 1

    return matrix, final_borders, selected_class_ids
