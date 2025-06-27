import sys
import typing as tp
from numpy.typing import NDArray

import numpy as np
import pandas as pd

sys.path.insert(0, '..')
from dsdna_mpra import config  # noqa E402


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


def merge_active_tiles_to_cres(
    paired_tiles: pd.DataFrame,
    tile_size: int,
) -> tp.Tuple[np.ndarray, np.ndarray]:

    def _filter_and_merge(lfc: pd.Series) -> np.ndarray:
        positions = paired_tiles[lfc > paired_tiles["threshold"]]["begin"].to_numpy(dtype=np.int64)
        if positions.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        intervals = np.stack([positions, positions + tile_size], axis=1)
        return merge(intervals)

    fwd_cres = _filter_and_merge(paired_tiles["fwd_lfc"])
    rev_cres = _filter_and_merge(paired_tiles["rev_lfc"])
    return fwd_cres, rev_cres


def merge_cres_across_strands(cre_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ['family', 'strain', 'genome', 'cell']
    merged_rows = []
    for group_values, group_df in cre_df.groupby(group_cols):
        fwd = group_df.loc[group_df['strand'] == '+', ['begin', 'end']].to_numpy()
        rev = group_df.loc[group_df['strand'] == '-', ['begin', 'end']].to_numpy()
        intervals = np.vstack([fwd, rev]) if fwd.size + rev.size > 0 else np.empty((0, 2), dtype=np.int64)
        merged = merge(intervals)
        merged_df = pd.DataFrame(merged, columns=['begin', 'end'])
        for col, val in zip(group_cols, group_values):
            merged_df[col] = val
        merged_rows.append(merged_df)
    result_df = pd.concat(merged_rows, ignore_index=True)
    result_df = result_df[group_cols + ['begin', 'end']]
    result_df.sort_values(by=group_cols + ['begin'], inplace=True)
    return result_df


def merge_cres_across_cells(cre_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ['family', 'strain', 'genome']
    merged_rows = []
    for group_vals, group_df in cre_df.groupby(group_cols):
        regions = group_df[['begin', 'end']].to_numpy()
        merged = merge(regions)
        merged_df = pd.DataFrame(merged, columns=['begin', 'end'])
        for col, val in zip(group_cols, group_vals):
            merged_df[col] = val
        merged_rows.append(merged_df)
    result_df = pd.concat(merged_rows, ignore_index=True)
    result_df = result_df[group_cols + ['begin', 'end']]
    result_df.sort_values(by=group_cols + ['begin'], inplace=True)
    return result_df


def main() -> None:
    # load and filter tile data
    virus_genomes = pd.read_csv(config.RAW_DIR / 'virus_genbank_ids.txt').columns.values
    paired_tiles = pd.read_csv(config.PROCESSED_DIR / 'virus_paired_tiles_log2p_ratios.csv')

    paired_tiles = paired_tiles[
        paired_tiles['genome'].isin(virus_genomes) &
        paired_tiles['family'].isin(config.DSDNA_FAMILIES)
    ].reset_index(drop=True)

    thresholds_df = pd.read_csv(config.RESULTS_DIR / 'thresholds_log2_1p.csv')[['cell', 'threshold']]
    paired_tiles = pd.merge(
        paired_tiles.fillna(0),
        thresholds_df,
        on='cell',
        how='left'
    )

    # collect CREs by strand
    cre_records = []
    for (cell, genome), virus_df in paired_tiles.groupby(['cell', 'genome']):
        family = virus_df['family'].iloc[0]
        strain = virus_df['strain'].iloc[0]

        fwd_cres, rev_cres = merge_active_tiles_to_cres(virus_df, tile_size=200)
        for strand, cres_array in zip(['+', '-'], [fwd_cres, rev_cres]):
            for begin, end in cres_array:
                cre_records.append([family, strain, cell, genome, strand, begin, end])

    cre_df = pd.DataFrame(cre_records, columns=['family', 'strain', 'cell', 'genome', 'strand', 'begin', 'end'])
    cre_df.sort_values(['family', 'strain', 'cell', 'genome', 'strand', 'begin'], inplace=True)
    cre_df.to_csv(config.RESULTS_DIR / 'cre_positions.csv', index=False)

    # merge across strands
    strand_merged_df = merge_cres_across_strands(cre_df)
    strand_merged_df.sort_values(['family', 'strain', 'cell', 'genome', 'begin'], inplace=True)
    strand_merged_df.to_csv(config.RESULTS_DIR / 'cre_positions_strands_merged.csv', index=False)

    # merge across cell lines
    cell_merged_df = merge_cres_across_cells(strand_merged_df)
    cell_merged_df.sort_values(['family', 'strain', 'genome', 'begin'], inplace=True)
    cell_merged_df.to_csv(config.RESULTS_DIR / 'cre_positions_strands_and_cell_merged.csv', index=False)


if __name__ == "__main__":
    main()
