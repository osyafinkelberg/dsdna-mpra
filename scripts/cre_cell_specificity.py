import typing as tp
import sys

import numpy as np
import pandas as pd
from numpy.typing import NDArray  # noqa F401

sys.path.insert(0, '..')
from dsdna_mpra import config  # noqa E402


def pad_matrix(
    matrix: tp.List[tp.List[tp.Optional[int]]],
    pad_value: float
) -> tp.List[tp.List[float]]:
    """
    Pads each sublist in a matrix to the length of the longest sublist using the pad_value.

    :param matrix: A 2D list with inner lists of unequal lengths.
    :param pad_value: Value used to pad the shorter sublists.
    :return: A 2D list with all rows of equal length.
    """
    max_len = max((len(row) for row in matrix), default=0)
    return [row + [pad_value] * (max_len - len(row)) for row in matrix]


def compute_coverage_matrix(
    intervals: NDArray[np.int64],
    total_length: int
) -> tp.Tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    Generates slice intervals and a coverage matrix indicating which input intervals overlap each slice.

    :param intervals: An (N, 2) array of [start, end) integer intervals.
    :param total_length: The total length of the sequence to cover.
    :return:
        - slices: A (P, 2) array of non-overlapping contiguous intervals.
        - coverage_matrix: A (P, T) array where each row contains indices
          of overlapping input intervals, padded with NaN.
    """
    if intervals.shape[0] == 0:
        return np.array([[0, total_length]], dtype=np.int64), np.array([[np.nan]], dtype=np.float64)

    boundaries = np.unique(
        np.concatenate([intervals[:, 0], intervals[:, 1], [total_length]]), sorted=True
    )
    slices = np.column_stack([boundaries[:-1], boundaries[1:]])
    coverage = [[] for _ in range(len(slices))]

    for interval_idx, (start, end) in enumerate(intervals):
        for i, (slice_start, slice_end) in enumerate(slices):
            if slice_start >= end:
                break
            if slice_end <= start:
                continue
            coverage[i].append(interval_idx)

    padded_coverage = pad_matrix(coverage, pad_value=np.nan)
    coverage_array = np.array(padded_coverage, dtype=np.float64)
    return slices, coverage_array


def main() -> None:
    genomes_summary = pd.read_csv(config.PROCESSED_DIR / 'summary_virus_genome_records.csv')
    cres_df = pd.read_csv(config.RESULTS_DIR / "cre_positions_strands_merged.csv")
    cres_df = cres_df.merge(
        genomes_summary[['accession_id', 'genome_size']].rename(columns={'accession_id': 'genome'}),
        on='genome',
        how='left'
    )

    partitioned_cres = []
    group_keys = ['family', 'strain', 'genome']
    for (family, strain, genome), group_df in cres_df.groupby(group_keys):
        genome_size = int(group_df['genome_size'].iloc[0])
        intervals = group_df[['begin', 'end']].values.astype(np.int64)
        slice_positions, coverage_indices = compute_coverage_matrix(intervals, genome_size)
        n_covering = (~np.isnan(coverage_indices)).sum(axis=1)

        # convert coverage indices to corresponding cell names
        cell_map = dict(enumerate(group_df['cell'].values))
        coverage_to_cell = np.vectorize(cell_map.get)
        covered_cells = coverage_to_cell(coverage_indices[n_covering != 0])

        # count cell overlaps per slice
        covered_slices = slice_positions[n_covering != 0]
        cell_counts = np.vstack([
            (covered_cells == cell_line).sum(axis=1) for cell_line in config.CELL_LINES
        ] + [n_covering[n_covering != 0]]).T

        result_df = pd.DataFrame(
            np.hstack([covered_slices, cell_counts]),
            columns=['begin', 'end'] + config.CELL_LINES + ['n_cells']
        )
        result_df.insert(0, 'family', family)
        result_df.insert(1, 'strain', strain)
        result_df.insert(2, 'genome', genome)
        partitioned_cres.append(result_df)

    final_df = pd.concat(partitioned_cres, ignore_index=True)
    output_path = config.RESULTS_DIR / 'cre_positions_partitioned_cres_strands_merged.csv'
    final_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
