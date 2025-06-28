import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '..')
from dsdna_mpra import config, clustering  # noqa E402


def main() -> None:
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
    paired_tiles['tile_lfc'] = paired_tiles[['fwd_lfc', 'rev_lfc']].max(1)
    cre_tiles_df = paired_tiles[paired_tiles.tile_lfc > paired_tiles.threshold].reset_index(drop=True)

    pivot_df = cre_tiles_df.pivot_table(
        index=['genome', 'begin'], columns='cell', values='tile_lfc', fill_value=0
    )
    tile_lfc_array = pivot_df.reindex(columns=config.CELL_LINES, fill_value=0).to_numpy()
    q_low = np.quantile(tile_lfc_array, 0.01)
    q_high = np.quantile(tile_lfc_array, 0.99)
    tile_lfc_array = np.clip(tile_lfc_array, q_low, q_high)

    # pairwise correlations across cell lines
    pcc_matrix = clustering.vectorized_pearsonr(tile_lfc_array, tile_lfc_array)
    np.save(config.RESULTS_DIR / 'cre_tiles_cell_lines_correlations.npy', pcc_matrix)

    # supervised clustering of binary activity patterns
    matrix, final_borders, selected_class_ids = clustering.supervised_binary_clustering(
        tile_lfc_array, coverage_threshold=0.8, binarization_threshold=0.5
    )
    np.savez(
        config.RESULTS_DIR / 'cre_tiles_supervised_clustering.npz',
        matrix=matrix,
        final_borders=final_borders,
        selected_class_ids=selected_class_ids
    )


if __name__ == "__main__":
    main()
