from typing import Dict, Optional, Callable
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from tqdm import tqdm
import json
import h5py

from . import config


"""
PWM Processing
"""


def pfm_to_ppm(
    count_matrix: np.ndarray,
    nucl_frequencies: np.ndarray = np.full(4, 0.25),
    pseudocount: float = 1e-2
) -> np.ndarray:
    """
    Converts a Position Frequency Matrix (PFM) to a Position Probability Matrix (PPM)
    using Laplace smoothing.

    Parameters:
        count_matrix (np.ndarray): Raw nucleotide counts (shape: 4 x N).
        nucl_frequencies (np.ndarray): Background nucleotide frequencies (A, C, G, T).
        pseudocount (float): Smoothing parameter, scaled by average total count per column.

    Returns:
        np.ndarray: Position Probability Matrix (PPM).

    Notes:
        - Background frequencies are used only for Laplace smoothing here.
        - Background normalization is performed in `ppm_to_pwm`, not here.
    """
    assert np.allclose(nucl_frequencies.sum(), 1, rtol=1e-10), \
        "Nucleotide frequencies must sum to 1"

    avg_col_sum = count_matrix.sum(axis=0).mean()
    scaled_pseudocount = pseudocount * avg_col_sum

    background = np.repeat(nucl_frequencies[:, np.newaxis], count_matrix.shape[1], axis=1)
    freq_matrix = count_matrix + scaled_pseudocount * background
    ppm = freq_matrix / freq_matrix.sum(axis=0)

    return ppm


def ppm_to_pwm(
    proba_matrix: np.ndarray,
    nucl_frequencies: np.ndarray = np.full(4, 0.25),
    indices: Optional[np.ndarray] = None,
    pseudocount: float = 0.0
) -> np.ndarray:
    """
    Converts a Position Probability Matrix (PPM) to a Position Weight Matrix (PWM),
    with optional pseudocount smoothing.

    Parameters:
        proba_matrix (np.ndarray): Normalized probabilities (shape: 4 x N).
        nucl_frequencies (np.ndarray): Background frequencies (A, C, G, T).
        indices (Optional[np.ndarray]): Optional column indices to use for PWM calculation.
        pseudocount (float): Value to add to each element in the PPM for smoothing. Default is 0.0.

    Returns:
        np.ndarray: Position Weight Matrix (PWM).
    """
    assert np.allclose(nucl_frequencies.sum(), 1, rtol=1e-10), \
        "Nucleotide frequencies must sum to 1"

    if indices is not None:
        proba_matrix = proba_matrix[:, indices]

    if pseudocount > 0:
        proba_matrix += pseudocount
        proba_matrix /= proba_matrix.sum(axis=0, keepdims=True)

    background = np.repeat(nucl_frequencies[:, np.newaxis], proba_matrix.shape[1], axis=1)
    energy = -np.log(proba_matrix) + np.log(background)
    energy -= energy.min(axis=0)

    return -energy


def pfm_to_pwm(
    count_matrix: np.ndarray,
    nucl_frequencies: np.ndarray = np.full(4, 0.25),
    pseudocount: float = 1e-2,
    indices: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Converts a Position Frequency Matrix (PFM) directly to a Position Weight Matrix (PWM).

    Parameters:
        count_matrix (np.ndarray): Raw nucleotide counts (shape: 4 x N).
        nucl_frequencies (np.ndarray): Background nucleotide frequencies.
        pseudocount (float): Smoothing parameter.
        indices (Optional[np.ndarray]): Subset of columns to use for PWM computation.

    Returns:
        np.ndarray: Position Weight Matrix (PWM).
    """
    ppm = pfm_to_ppm(count_matrix, nucl_frequencies, pseudocount)
    pwm = ppm_to_pwm(ppm, nucl_frequencies, indices)
    return pwm


def pfm_to_cwm(pfm: np.ndarray) -> np.ndarray:
    pwm = pfm_to_pwm(pfm)
    return 2 * (pwm - pwm.mean(0)) / np.abs(pwm).sum(0).max()


"""
Input / Output
"""


def parse_pwm_file(pwm_path: Path) -> Dict[str, np.ndarray]:
    """
    Parses a file containing multiple Position Weight Matrices (PWMs).

    Args:
        pwm_path (Path): Path to the file containing PWMs in .cb format.

    Returns:
        Dict[str, np.ndarray]: A dictionary mapping PWM identifiers to their
                               corresponding PWM matrix (as a NumPy array, transposed).
    """
    pwm_dict = {}
    pwm_id = ''
    pwm_rows = []
    with pwm_path.open('r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            if line.startswith('>'):
                if pwm_id:
                    pwm_dict[pwm_id] = np.array(pwm_rows, dtype=float).T
                pwm_id = line.strip().removeprefix('>')
                pwm_rows = []
            else:
                pwm_rows.append(np.fromstring(line.strip(), sep=' '))
        # save the last PWM
        if pwm_id:
            pwm_dict[pwm_id] = np.array(pwm_rows, dtype=float).T
    return pwm_dict


def write_pwm_file(pwm_dict: Dict[str, np.ndarray], pwm_path: Path) -> None:
    """
    Writes Position Weight Matrices (PWMs) to a file in .cb format.

    Args:
        pwm_dict (Dict[str, np.ndarray]): Dictionary mapping PWM identifiers
                                          to their PWM matrices (NumPy arrays).
        pwm_path (Path): Path to the output .cb file.
    """
    with pwm_path.open('w') as file:
        for pwm_id, pwm_matrix in pwm_dict.items():
            file.write(f'>{pwm_id}\n')
            # ensure the matrix is transposed back to match original row layout
            for row in pwm_matrix.T:
                row_str = ' '.join(f'{val:.6f}' for val in row)
                file.write(f'{row_str}\n')


"""
Motif-to-Contribution Score Match
"""


def rolling_absolute_contribution_scores(scores: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Calculate rolling total absolute contribution scores from a 2D array of scores.

    Args:
        scores (np.ndarray): 2D array of contribution scores.
        window (int): Rolling window size.

    Returns:
        np.ndarray: Array of rolling total absolute contribution scores.
    """
    negative_sums = -scores.clip(max=0).sum(axis=0)
    positive_sums = scores.clip(min=0).sum(axis=0)
    total_abs_contrib = np.maximum(negative_sums, positive_sums)
    cumsum = np.cumsum(total_abs_contrib)
    half_window_left = (window + 1) // 2
    half_window_right = window // 2
    total_abs_contrib[half_window_left: -half_window_right] = (
        cumsum[window:] - cumsum[:-window]
    )
    return total_abs_contrib


def suppress_non_maximum_peaks(
    peaks: np.ndarray, peak_scores: np.ndarray,
    radius: int, overlap_threshold: float
) -> np.ndarray:
    """
    Given sorted peak positions and their scores, iteratively suppress
    weaker peaks that overlap stronger neighbors by more than overlap_threshold.
    """
    active = np.pad(np.full(peaks.size, True), (1, 1), mode='constant')
    for idx in peak_scores.argsort():
        left_idx = idx - np.argmax(active[:idx + 1][::-1])
        right_idx = idx + 2 + np.argmax(active[idx + 2:])
        peak_pos = peaks[idx]
        left_pos = peaks[left_idx - 1] if active[left_idx] else -2 * radius
        right_pos = peaks[right_idx - 1] if active[right_idx] else 200 + 2 * radius
        overlap = 2 * radius - min(peak_pos - left_pos, right_pos - peak_pos)
        if overlap > 2 * radius * overlap_threshold:
            active[idx + 1] = False
    return peaks[active[1:-1]]


def call_absolute_contribution_peaks(
    tacs: np.ndarray,
    summit_threshold: float,
    peak_radius: int,
    overlap_threshold: float
) -> np.ndarray:
    increasing = np.pad(tacs[1:] - tacs[:-1] >= 0, (1, 0), mode='constant')
    decreasing = np.pad(tacs[1:] - tacs[:-1] <= 0, (0, 1), mode='constant')
    local_max_mask = increasing & decreasing
    threshold_mask = tacs > summit_threshold
    candidate_peaks = np.argwhere(local_max_mask & threshold_mask).flatten()
    peak_summits = suppress_non_maximum_peaks(candidate_peaks, tacs[candidate_peaks], peak_radius, overlap_threshold)

    peaks = np.repeat(peak_summits.reshape(-1, 1), 2, axis=1)
    boundaries = peaks + np.array([-peak_radius, peak_radius])
    return boundaries.clip(0, 199)


def rolling_continuous_jaccard_similarity(
    scores: np.ndarray,
    motif_weights: np.ndarray,
    motif_shift: int
) -> np.ndarray:
    """
    Compute rolling Continuous Jaccard Similarity (CJS) between scores and motif weights.

    The Continuous Jaccard Similarity metric was introduced by Avanti Shrikumar et al. (2020)
    (arXiv:1811.00416). This function implements an efficient computational approach.

    Parameters:
        scores (np.ndarray): Input contribution score array.
        motif_weights (np.ndarray): Motif weight matrix.
        motif_shift (int): Position of maximum information content in the motif.

    Returns:
        np.ndarray: Rolling CJS values, padded to match input length.
    """
    windows = sliding_window_view(scores, window_shape=(4, motif_weights.shape[1]), axis=(0, 1))[0]
    norm_motif = motif_weights / np.abs(motif_weights).sum()
    norm_windows = windows / np.abs(windows).sum(axis=(1, 2))[:, None, None]
    intersection = np.minimum(np.abs(norm_motif), np.abs(norm_windows)) * np.sign(motif_weights) * np.sign(windows)
    union = np.maximum(np.abs(norm_motif), np.abs(norm_windows))
    similarity = intersection.sum(axis=(1, 2)) / union.sum(axis=(1, 2))
    pad_left = motif_shift
    pad_right = motif_weights.shape[1] - motif_shift - 1
    return np.pad(similarity, (pad_left, pad_right), mode='constant')  # motif max IC position


def batch_rolling_cjs(
    scores: np.ndarray,
    motif_cwms: dict[str, np.ndarray],
    motif_shifts: dict[str, int]
) -> np.ndarray:
    motifs = sorted(motif_cwms.keys())
    cjs_matrix = np.zeros((len(motifs), scores.shape[1]))
    for i, motif in enumerate(motifs):
        cwm = motif_cwms[motif]
        shift = motif_shifts[motif]
        cjs_matrix[i] = rolling_continuous_jaccard_similarity(scores, cwm, shift)
    return cjs_matrix


def find_best_motif_matches(
    cjs_matrix: np.ndarray,
    motifs: np.ndarray,
    motif_shifts: np.ndarray,
    peak_summits: np.ndarray,
    peak_radius: int,
    top_k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window_size = 2 * peak_radius
    # extract sliding windows around each peak summit; shape: (num_peaks, num_motifs, window_size)
    peak_windows = sliding_window_view(cjs_matrix, window_shape=(len(motifs), window_size), axis=(0, 1))[0]

    # calculate safe start indices for slicing peak_windows along axis=0 to avoid underflow
    start_indices = peak_summits - np.minimum(peak_summits, peak_radius)
    peak_windows = peak_windows[start_indices, :, :]

    # find best motif match index and score per position within each peak window
    best_match_idx = peak_windows.argmax(axis=1)  # shape: (num_peaks, window_size)
    best_scores = peak_windows.max(axis=1)  # shape: (num_peaks, window_size)

    # for each peak, get top_k best scoring positions (indices within the window)
    # gather the corresponding best motif indices for the top positions
    top_positions_idx = np.argsort(best_scores, axis=1)[:, -top_k:][:, ::-1]
    num_peaks = peak_summits.size
    repeated_peak_idx = np.arange(num_peaks).repeat(top_k)
    flat_positions = top_positions_idx.ravel()
    top_motif_matches = best_match_idx[repeated_peak_idx, flat_positions].reshape(num_peaks, top_k)

    # calculate actual genomic positions adjusted for peak_radius and motif shifts
    # clip positions to valid range (assuming length 200 as per previous functions)
    adjusted_positions = (
        top_positions_idx + peak_summits[:, None] - peak_radius - motif_shifts[top_motif_matches]
    )
    clipped_positions = adjusted_positions.clip(0, 199)

    # sort top scores for output to align with matches
    sorted_scores = np.take_along_axis(best_scores, top_positions_idx, axis=1)
    return clipped_positions, sorted_scores, motifs[top_motif_matches]


def build_contribution_score_dataframes(
    match_positions: np.ndarray,
    matched_motifs: np.ndarray,
    cwms: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, ...]:
    """
    Convert best motif matches and their positions into a list of
    DataFrames representing contribution score arrays.

    Each DataFrame has shape (200, 4) with columns ['A', 'C', 'G', 'T'].
    """
    cs_dfs = []
    num_matches = matched_motifs.shape[1]
    seq_length = 200
    for i in range(num_matches):
        cs_array = np.zeros((4, seq_length), dtype=np.float64)
        for peak_idx in range(matched_motifs.shape[0]):
            motif = matched_motifs[peak_idx, i]
            cwm = cwms[motif]
            pos = match_positions[peak_idx, i, 0]
            cs_array[:, pos: pos + cwm.shape[1]] = cwm
        cs_dfs.append(pd.DataFrame(cs_array.T, columns=['A', 'C', 'G', 'T']))
    return tuple(cs_dfs)


"""
Utility functions for large-scale analysis.
"""


def compute_jaccard_similarity_scores(
    tf_motifs: dict[str, np.ndarray],
    cwm_shifts: dict[str, int],
    contrib_scores: np.ndarray,
    tile_ids: np.ndarray,
    output_path: Path
) -> None:

    num_tiles, num_motifs = contrib_scores.shape[0], len(tf_motifs)
    cjs_arr = np.zeros((num_tiles, num_motifs, 200))
    with tqdm(total=num_tiles, desc='jaccard similarity', dynamic_ncols=True, leave=False) as pbar:
        for tile_idx, cs_arr in enumerate(contrib_scores):
            cjs_arr[tile_idx] = batch_rolling_cjs(cs_arr, tf_motifs, cwm_shifts)

            if (tile_idx + 1) % 10000 == 0 or tile_idx + 1 == num_tiles:
                step = 10000 if tile_idx + 1 < num_tiles else num_tiles % 10000
                pbar.update(step)

    with h5py.File(
        output_path, "w"
    ) as f:
        _ = f.create_dataset("continuous_jaccard_similarity", data=cjs_arr, dtype='f')
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset("tile_ids", data=tile_ids.astype(dt))
        f.create_dataset("motif_ids", data=np.array(sorted(tf_motifs.keys()), dtype=dt))


def match_motifs_to_contribution_score_peaks(
    tf_motifs: dict[str, np.ndarray],
    cwm_shifts_unordered: dict[str, int],
    contrib_scores: np.ndarray,
    cjs_path: Path,
    output_path: Path
) -> None:

    cs_peaks = []
    num_tiles = contrib_scores.shape[0]
    with tqdm(total=num_tiles, desc='tiles peak calling', dynamic_ncols=True, leave=False) as pbar:
        for tile_idx in range(num_tiles):
            cs_arr = contrib_scores[tile_idx]
            tacs = rolling_absolute_contribution_scores(cs_arr, window=config.TACS_WINDOW)
            peaks = call_absolute_contribution_peaks(
                tacs=tacs,
                summit_threshold=config.PER_POS_THRESHOLD * config.TACS_WINDOW,
                peak_radius=config.PEAK_RADIUS,
                overlap_threshold=config.OVERLAP_THRESHOLD
            )
            matched_tacs = tacs[peaks.mean(1).astype(int)]
            cs_peaks.append((peaks, matched_tacs))

            if (tile_idx + 1) % 10000 == 0 or tile_idx + 1 == num_tiles:
                step = 10000 if tile_idx + 1 < num_tiles else num_tiles % 10000
                pbar.update(step)

    with h5py.File(cjs_path, "r") as f:
        cjs_arr = f["continuous_jaccard_similarity"][()]
        tile_ids = f["tile_ids"][()].astype(str)
        motif_ids = f["motif_ids"][()].astype(str)

    cwm_shifts = np.array([cwm_shifts_unordered[motif_id] for motif_id in motif_ids])

    best_match = []
    with tqdm(total=num_tiles, desc='map top-1 CJS match', dynamic_ncols=True, leave=False) as pbar:
        for tile_idx, (peak_positions, match_tacs) in enumerate(cs_peaks):
            if not peak_positions.size:
                best_match.append({
                    "tile_id": tile_ids[tile_idx],
                    "peak_positions": [],
                    "motif_positions": [],
                    "motifs": [],
                    "tacs": [],
                    "cjs": []
                })
                continue

            match_pos, match_cjs, match_motif = find_best_motif_matches(
                cjs_arr[tile_idx],
                motif_ids,
                cwm_shifts,
                peak_positions.mean(1).astype(np.uint8),
                config.PEAK_RADIUS,
                top_k=1
            )

            match_positions = [
                [pos, pos + tf_motifs[motif].shape[1]]
                for pos, motif in zip(match_pos.squeeze(1).tolist(), match_motif.squeeze(1).tolist())
            ]

            best_match.append({
                "tile_id": tile_ids[tile_idx],
                "peak_positions": peak_positions.astype(int).tolist(),
                "motif_positions": match_positions,
                "motifs": match_motif.squeeze(1).tolist(),
                "tacs": match_tacs.tolist(),
                "cjs": match_cjs.squeeze(1).tolist()
            })

            if (tile_idx + 1) % 10000 == 0 or tile_idx + 1 == num_tiles:
                step = 10000 if tile_idx + 1 < num_tiles else num_tiles % 10000
                pbar.update(step)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(best_match, f, ensure_ascii=False, indent=1)


def compute_tfbs_counts(
    input_df: pd.DataFrame,
    id_func: Callable[[pd.Series], str],
    output_path: Path,
    base_columns: list[str],
    tile_motif_map: dict[str, dict],
    motif_to_gene_map: dict[str, str]
) -> None:
    """
    Compute transcription factor binding site (TFBS) counts per element and save the result.

    For each row in the input DataFrame, identifies associated motifs using the provided
    `tile_motif_map` and maps them to transcription factor (TF) genes using `motif_to_gene_map`.
    Counts the occurrences of each TF in the K562 cell line, excluding specified TFs,
    and appends these counts to the base columns. Saves the resulting DataFrame to a CSV file.

    Args:
        input_df: Input DataFrame containing genomic elements.
        id_func: Function to extract a unique ID for each row (used to query tile_motif_map).
        output_path: Path to save the output CSV file with TFBS counts.
        base_columns: List of column names to retain from the input DataFrame.
        tile_motif_map: Dictionary mapping element IDs to motif data.
        motif_to_gene_map: Dictionary mapping motif names to TF gene names.
    """
    tfbs_counts = []

    for row in input_df[base_columns].itertuples(index=False):
        element_id = id_func(row)
        element_info = tile_motif_map.get(element_id, {})
        motifs = element_info.get('motifs', [])
        element_tfbs_counts = [0] * len(config.TF_GENES_K562)
        for motif in motifs:
            cleaned_motif = motif.removesuffix('_fwd').removesuffix('_rev')
            tf_genes_str = motif_to_gene_map.get(cleaned_motif, "")
            for gene in tf_genes_str.split('-'):
                if gene in config.TF_GENES_K562_EXCLUDED:
                    continue
                idx = config.TF_GENES_K562_INDEX.get(gene)
                if idx is not None:
                    element_tfbs_counts[idx] += 1

        tfbs_counts.append(list(row) + element_tfbs_counts)

    df = pd.DataFrame(tfbs_counts, columns=base_columns + config.TF_GENES_K562)
    df.to_csv(output_path, index=False)
