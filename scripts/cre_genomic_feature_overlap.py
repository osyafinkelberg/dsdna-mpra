import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '..')
from dsdna_mpra import config, clustering  # noqa E402


def compute_cell_rankings(paired_tiles: pd.DataFrame) -> pd.DataFrame:
    ranks = []
    for cell, group in paired_tiles.groupby('cell'):
        group = group.copy()
        group['cell_rank'] = group.tile_lfc.rank(method='first') / len(group)
        ranks.append(group[['cell', 'genome', 'begin', 'cell_rank']])
    rank_df = pd.concat(ranks, ignore_index=True)
    return pd.merge(paired_tiles, rank_df, on=['cell', 'genome', 'begin'], how='left')


def mark_cres_whose_center_falls_in_regions(
    cres_df: pd.DataFrame,
    regions_df: pd.DataFrame,
    overlap_col_name: str
) -> pd.DataFrame:
    unique_cres_df = (
        cres_df
        .drop_duplicates(['genome', 'begin', 'end'])[['genome', 'begin', 'end']]
        .reset_index(drop=True)
    )
    marked_cres = []
    for genome, genome_cres in unique_cres_df.groupby('genome'):
        cres = genome_cres[['begin', 'end']].values.mean(1).astype(int)
        regions = regions_df[regions_df.genome == genome][['begin', 'end']].values
        overlaps = ((cres[:, None] >= regions[:, 0]) & (cres[:, None] <= regions[:, 1])).any(axis=1)
        genome_cres = genome_cres.copy()
        genome_cres[overlap_col_name] = overlaps
        marked_cres.append(genome_cres)
    overlap_df = pd.concat(marked_cres, ignore_index=True)
    return pd.merge(cres_df, overlap_df, on=['genome', 'begin', 'end'], how='left')


def mark_regions_that_contain_cre_centers(
    cres_df: pd.DataFrame,
    regions_df: pd.DataFrame,
    overlap_col_name: str
) -> pd.DataFrame:
    marked_regions = []
    for genome, genome_regions in regions_df.groupby('genome'):
        region_bounds = genome_regions[['begin', 'end']].values
        cre_centers = cres_df[cres_df.genome == genome][['begin', 'end']].values.mean(1).astype(int)
        overlaps = [
            ((cre_centers >= region_begin) & (cre_centers <= region_end)).any()
            for region_begin, region_end in region_bounds
        ]
        genome_regions = genome_regions.copy()
        genome_regions[overlap_col_name] = overlaps
        marked_regions.append(genome_regions)
    overlap_df = pd.concat(marked_regions, ignore_index=True)
    return pd.merge(regions_df, overlap_df, on=['genome', 'begin', 'end'], how='left')


def add_max_tile_value_to_regions(
    tiles_df: pd.DataFrame,
    regions_df: pd.DataFrame,
    tile_value_col_name: str
) -> pd.DataFrame:
    results = []
    for genome, genome_regions in regions_df.groupby('genome'):
        genome_tiles = tiles_df[tiles_df['genome'] == genome].copy()
        genome_tiles['center'] = genome_tiles[['begin', 'end']].mean(axis=1).astype(int)
        region_bounds = genome_regions[['begin', 'end']].values
        for cell in config.CELL_LINES:
            cell_tiles = genome_tiles[genome_tiles['cell'] == cell]
            max_vals = []
            for region_begin, region_end in region_bounds:
                overlapping_tiles = cell_tiles[
                    (cell_tiles['center'] >= region_begin) &
                    (cell_tiles['center'] <= region_end)
                ]
                if not overlapping_tiles.empty:
                    tile_value_max = overlapping_tiles[tile_value_col_name].max()
                else:
                    tile_value_max = float('nan')
                max_vals.append(tile_value_max)
            cell_regions = genome_regions.copy()
            cell_regions['cell'] = cell
            cell_regions[f'max_{tile_value_col_name}'] = max_vals
            results.append(cell_regions)
    return pd.concat(results, ignore_index=True)


def compute_promoter_activity_per_kinetic_group(
    kinetics_df: pd.DataFrame,
    tile_value_col_name: str
) -> pd.DataFrame:
    stats = list()
    for strain, strain_df in kinetics_df.groupby('strain'):
        strain_df.cell = strain_df.cell.astype('category').cat.set_categories(config.CELL_LINES)
        for cell, cell_df in strain_df.groupby('cell', observed=False):
            cell_lst = [strain, cell]
            for kinetic_group in config.GENE_KINETIC_GROUPS:
                group_df = cell_df[cell_df[kinetic_group]]
                cell_lst.append(group_df[f'max_{tile_value_col_name}'].mean())
            stats.append(cell_lst)
    return pd.DataFrame(stats, columns=['strain', 'cell'] + config.GENE_KINETIC_GROUPS)


def compute_observed_expected_overlap(
    regions_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    overlap_col_name: str
) -> pd.DataFrame:
    fractions_df = list()
    group_columns = ['family', 'genome', 'genome_size']
    for (family, genome, genome_size), overlap_df in overlap_df.groupby(group_columns):
        # Observed: average across cell lines
        observed = overlap_df.groupby('cell').mean(overlap_col_name)[overlap_col_name].mean()
        # Genomic regions (e.g., CDSs) may appear to overlap themselves
        genome_regions = regions_df[regions_df.genome == genome]
        merged_regions = clustering.merge(np.sort(genome_regions[['begin', 'end']].values, axis=1))
        expected = (merged_regions[:, 1] - merged_regions[:, 0]).sum() / genome_size
        fractions_df.append({'family': family, 'genome': genome, 'observed': observed, 'expected': expected})
    return fractions_df


def main() -> None:

    # 250 bp range surrounding TSS
    PROXIMITY_RANGE = 250

    # load MPRA data
    genomes_summary = pd.read_csv(config.PROCESSED_DIR / 'summary_virus_genome_records.csv')
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
    paired_tiles['end'] = paired_tiles.begin + 200  # tiles are 200 bp

    # overlap between tiles and CDSs
    cds_df = pd.read_csv(config.PROCESSED_DIR / 'virus_cds_positions.csv')
    paired_tiles = compute_cell_rankings(paired_tiles)
    paired_tiles_cds = mark_cres_whose_center_falls_in_regions(paired_tiles, cds_df, 'is_cds')
    paired_tiles_cds.to_csv(config.RESULTS_DIR / "virus_paired_tiles_cds_overlap.csv", index=False)

    # overlap between CREs and CDSs
    cres_df = pd.merge(
        pd.read_csv(config.RESULTS_DIR / "cre_positions_strands_merged.csv"),
        genomes_summary[['accession_id', 'genome_size']].rename({'accession_id': 'genome'}, axis=1),
        on='genome', how='left'
    )
    cres_cds_df = mark_cres_whose_center_falls_in_regions(cres_df, cds_df, 'is_cds')
    cres_cds_df.to_csv(config.RESULTS_DIR / "cres_cds_overlap.csv", index=False)

    # observed and expected fractions of CREs in CDSs
    cres_cds_fractions_df = compute_observed_expected_overlap(cds_df, cres_cds_df, 'is_cds')
    pd.DataFrame(cres_cds_fractions_df).to_csv(config.RESULTS_DIR / "cres_in_cds_fractions.csv", index=False)

    # CREs proximal to GenBank gene start positions (observed and expected)
    gene_df = pd.read_csv(config.PROCESSED_DIR / 'virus_gene_positions.csv')
    gene_df['begin'] = gene_df.five_prime - PROXIMITY_RANGE
    gene_df['end'] = gene_df.five_prime + PROXIMITY_RANGE
    cres_tss_df = mark_cres_whose_center_falls_in_regions(cres_df, gene_df, 'proximal_to_gene_start')
    cres_tss_df.to_csv(config.RESULTS_DIR / "cres_proximal_to_gene_starts.csv", index=False)
    cres_tss_fractions_df = compute_observed_expected_overlap(gene_df, cres_tss_df, 'proximal_to_gene_start')
    pd.DataFrame(cres_tss_fractions_df).to_csv(
        config.RESULTS_DIR / "cres_proximal_to_gene_starts_fractions.csv", index=False
    )

    # GenBank gene start positions proximal to CREs (observed and expected)
    CELL = 'HEK293'
    tss_cres_df = mark_regions_that_contain_cre_centers(
        cres_df[cres_df.cell == CELL],
        gene_df[gene_df.genome.isin(virus_genomes)].reset_index(drop=True),
        'proximal_to_cre'
    )
    tss_cres_df = pd.merge(
        tss_cres_df,
        genomes_summary.rename({'accession_id': 'genome'}, axis=1)[['family', 'genome', 'genome_size']],
        on='genome', how='left'
    )
    tss_cres_df = tss_cres_df[tss_cres_df.family.notna()].reset_index(drop=True)
    tss_cres_df.to_csv(config.RESULTS_DIR / f"gene_starts_proximal_to_cres_{CELL.lower()}.csv", index=False)
    tss_cres_fractions_df = list()
    group_columns = ['family', 'genome', 'genome_size']
    for (family, genome, genome_size), overlap_df in tss_cres_df.groupby(group_columns):
        observed = overlap_df['proximal_to_cre'].mean()
        n_cres = cres_df[(cres_df.genome == genome) & (cres_df.cell == CELL)].shape[0]
        expected = 1 - (1 - 2 * PROXIMITY_RANGE / genome_size) ** n_cres
        tss_cres_fractions_df.append({'family': family, 'genome': genome, 'observed': observed, 'expected': expected})
    pd.DataFrame(tss_cres_fractions_df).to_csv(
        config.RESULTS_DIR / f"gene_starts_proximal_to_cres_fractions_{CELL.lower()}.csv", index=False
    )

    # overlap between tiles and CAGE-seq peaks
    CAGE_PROXIMITY_RANGE = 100
    cage_df = pd.concat([
        pd.read_csv(config.PROCESSED_DIR / 'cage_pmid_32341360_gbid_BK012101.1.csv'),
        pd.read_csv(config.PROCESSED_DIR / 'cage_pmid_33024035_gbid_NC_001348.1.csv'),
        pd.read_csv(config.PROCESSED_DIR / 'cage_pmid_29864140_gbid_V01555.2.csv'),
        pd.read_csv(config.PROCESSED_DIR / 'cage_pmid_38206015_gbid_GQ994935.1.csv'),
    ]).reset_index(drop=True)
    cage_df['begin'] = cage_df.five_prime - CAGE_PROXIMITY_RANGE
    cage_df['end'] = cage_df.five_prime + CAGE_PROXIMITY_RANGE
    cage_tiles = pd.read_csv(config.PROCESSED_DIR / 'virus_paired_tiles_log2p_ratios_cage_genomes.csv')
    cage_tiles['end'] = cage_tiles.begin + 200  # tiles are 200 bp
    cage_tiles['tile_lfc'] = cage_tiles[['fwd_lfc', 'rev_lfc']].max(1)
    cage_tiles = compute_cell_rankings(cage_tiles)
    cage_tiles = mark_cres_whose_center_falls_in_regions(cage_tiles, cds_df, 'is_cds')
    cage_tiles = mark_cres_whose_center_falls_in_regions(cage_tiles, cage_df, 'is_cage_peak')
    cage_tiles.to_csv(config.RESULTS_DIR / "virus_paired_tiles_cage_peaks_overlap.csv", index=False)

    TILE_FEATURE = 'cell_rank'

    # the most active tiles in proximity of gene starts in herpesvirus kinetic groups
    herpes_kinetics_df = pd.read_csv(config.RAW_DIR / 'herpesvirus_tss_kinetics_manual_annotation.csv')
    herpes_kinetics_df['begin'] = herpes_kinetics_df.five_prime - PROXIMITY_RANGE
    herpes_kinetics_df['end'] = herpes_kinetics_df.five_prime + PROXIMITY_RANGE
    herpes_kinetics_df = add_max_tile_value_to_regions(paired_tiles, herpes_kinetics_df, TILE_FEATURE)
    herpes_kinetics_df.to_csv(config.RESULTS_DIR / 'herpesvirus_tss_kinetics_cell_ranks.csv', index=False)
    compute_promoter_activity_per_kinetic_group(herpes_kinetics_df, TILE_FEATURE).to_csv(
        config.RESULTS_DIR / 'herpesvirus_tss_kinetics_average_cell_rank.csv', index=False
    )

    # Epstein Barr Virus (HHV-4) kinetic groups reported by Reza Djavadian et al.
    hhv4_kinetics = pd.read_csv(config.PROCESSED_DIR / 'cage_pmid_29864140_gbid_V01555.2_kinetics.csv')
    hhv4_kinetics['begin'] = hhv4_kinetics.five_prime - PROXIMITY_RANGE
    hhv4_kinetics['end'] = hhv4_kinetics.five_prime + PROXIMITY_RANGE
    hhv4_kinetics = add_max_tile_value_to_regions(cage_tiles, hhv4_kinetics, TILE_FEATURE)
    hhv4_kinetics.to_csv(
        config.RESULTS_DIR / 'hhv4_cage_pmid_29864140_kinetics_cell_ranks.csv', index=False
    )


if __name__ == "__main__":
    main()
