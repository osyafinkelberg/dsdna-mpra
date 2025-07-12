import typing as tp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import json

sys.path.insert(0, '..')
from dsdna_mpra import config, clustering  # noqa E402
from cre_genomic_feature_overlap import mark_cres_whose_center_falls_in_regions  # noqa E402


EXCLUDED_GENES = {"?", "3'SS"}
TF_GENE_INDEX = {gene: idx for idx, gene in enumerate(config.TF_GENES_K562)}


def compute_tfbs_counts(
    input_df: pd.DataFrame,
    id_func: tp.Callable[[pd.Series], str],
    output_path: Path,
    base_columns: list[str],
    tile_motif_map: dict[str, dict],
    motif_to_gene_map: dict[str, str]
) -> None:
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
                if gene in EXCLUDED_GENES:
                    continue
                idx = TF_GENE_INDEX.get(gene)
                if idx is not None:
                    element_tfbs_counts[idx] += 1

        tfbs_counts.append(list(row) + element_tfbs_counts)

    df = pd.DataFrame(tfbs_counts, columns=base_columns + config.TF_GENES_K562)
    df.to_csv(output_path, index=False)


def main() -> None:
    # load data
    virus_genomes = pd.read_csv(config.RAW_DIR / 'virus_genbank_ids.txt').columns.values
    thresholds_df = pd.read_csv(config.RESULTS_DIR / 'thresholds_log2_1p.csv')[['cell', 'threshold']]
    k562_threshold = thresholds_df.loc[thresholds_df['cell'] == 'K562', 'threshold'].iloc[0]

    malinois_data = np.load(config.RESULTS_DIR / "malinois_predicted_activities.npz", allow_pickle=True)
    dhs_tiles = malinois_data['dhs_tiles']
    active_dhs_ids = malinois_data['dhs_tile_ids'][dhs_tiles[:, 0] >= k562_threshold]
    malinois_df = pd.DataFrame({
        'tile_id': np.concatenate([malinois_data['virus_tile_ids'], malinois_data['dhs_tile_ids']]),
        'malinois_k562_lfc': np.concatenate([malinois_data['virus_tiles'][:, 0], malinois_data['dhs_tiles'][:, 0]])
    })
    with open(config.RESULTS_DIR / "malinois_K562_tf_motif_map.json", 'r', encoding='utf-8') as f:
        tile_motif_map = {entry['tile_id']: entry for entry in json.load(f)}
    motif_to_gene_map = dict(pd.read_csv(config.RESULTS_DIR / 'malinois_K562_tf_motif_families.csv').values)

    paired_tiles = pd.read_csv(config.PROCESSED_DIR / 'virus_paired_tiles_log2p_ratios.csv')
    paired_tiles = paired_tiles[
        paired_tiles['genome'].isin(virus_genomes) &
        paired_tiles['family'].isin(config.DSDNA_FAMILIES) &
        (paired_tiles['cell'] == 'K562')
    ].reset_index(drop=True)
    paired_tiles = paired_tiles.merge(malinois_df, on='tile_id', how='left')
    paired_tiles['virus'] = paired_tiles['family'] + ', ' + paired_tiles['strain']
    paired_tiles.to_csv(config.RESULTS_DIR / "malinois_predicted_and_observed_activities.csv", index=False)
    active_tiles = paired_tiles[paired_tiles['malinois_k562_lfc'] >= k562_threshold].reset_index(drop=True)

    dhs_df = pd.read_csv(config.RAW_DIR / 'K562-DS15363_peaks_with_sequences.bed', sep=',')
    dhs_df['tile_id'] = dhs_df.chromosome + '-' + dhs_df.center.astype(str)
    dhs_df = dhs_df.merge(malinois_df, on='tile_id', how='left')

    # number of TFBS per active tile
    active_tiles['n_tfbs'] = active_tiles['tile_id'].map(
        lambda tid: len(tile_motif_map.get(tid, {}).get('peak_positions', []))
    )
    active_tiles['tfbs_bucket'] = active_tiles['n_tfbs'].clip(upper=7).astype(str)
    active_tiles.loc[active_tiles['tfbs_bucket'] == '7', 'tfbs_bucket'] = '7+'
    counts = active_tiles.groupby(['virus', 'tfbs_bucket']).size().unstack(fill_value=0)
    counts = counts[[str(i) for i in range(7)] + ['7+']]
    counts = counts.reindex(config.VIRUSES, fill_value=0)
    n_tfbs_dhs = np.array([
        len(tile_motif_map.get(tile_id, {}).get('peak_positions', []))
        for tile_id in active_dhs_ids
    ])
    dhs_buckets = np.clip(n_tfbs_dhs, 0, 7).astype(str)
    dhs_buckets[dhs_buckets == '7'] = '7+'
    dhs_counts = pd.Series(dhs_buckets).value_counts().reindex(counts.columns, fill_value=0)
    counts.loc['DHS'] = dhs_counts
    counts.to_csv(config.RESULTS_DIR / "malinois_K562_number_tfbs_per_active_tile.csv", index=True)

    # TFBS counts for individual TFs
    tile_columns = ['tile_id', 'virus', 'genome', 'begin', 'fwd_lfc', 'rev_lfc', 'malinois_k562_lfc']
    compute_tfbs_counts(
        input_df=paired_tiles,
        id_func=lambda row: row.tile_id,
        output_path=config.RESULTS_DIR / "malinois_K562_tfbs_counts_virus_tiles.csv",
        base_columns=tile_columns,
        tile_motif_map=tile_motif_map,
        motif_to_gene_map=motif_to_gene_map
    )
    dhs_columns = ['chromosome', 'center', 'start', 'stop', 'intensity', 'malinois_k562_lfc']
    compute_tfbs_counts(
        input_df=dhs_df,
        id_func=lambda row: f'{row.chromosome}-{row.center}',
        output_path=config.RESULTS_DIR / "malinois_K562_tfbs_counts_dhs.csv",
        base_columns=dhs_columns,
        tile_motif_map=tile_motif_map,
        motif_to_gene_map=motif_to_gene_map
    )

    # TFBS counts in proximity to Herpesvirus promoters from different kinetic groups
    PROXIMITY_RANGE = 250  # 250 bp around TSS
    herpes_kinetics_df = pd.read_csv(config.RAW_DIR / 'herpesvirus_tss_kinetics_manual_annotation.csv')
    motif_map_df = pd.read_csv(config.RESULTS_DIR / 'malinois_K562_tf_motif_genome_annotation.csv')
    motif_map_df['center'] = motif_map_df[['begin', 'end']].mean(axis=1).astype(int)

    all_tss_with_tfbs = []
    for genome, tss_df in herpes_kinetics_df.groupby('genome', sort=False):
        motif_df = motif_map_df[motif_map_df.genome == genome]
        distances = tss_df.five_prime.values[:, None] - motif_df.center.values[None, :]
        proximity_mask = np.abs(distances) <= PROXIMITY_RANGE
        proximity_df = pd.DataFrame(
            np.argwhere(proximity_mask),
            columns=['tss_index', 'motif_index']
        )
        tfbs_counts_per_tss = []
        for tss_index in range(len(tss_df)):
            nearby_motifs = proximity_df[proximity_df.tss_index == tss_index]
            tfbs_counts = [0] * len(config.TF_GENES_K562)
            if not nearby_motifs.empty:
                tf_genes = motif_df.iloc[nearby_motifs.motif_index.values].tf_gene
                for tf_gene in tf_genes:
                    for gene in tf_gene.split('-'):  # Multiple TFs may be listed with hyphens
                        idx = TF_GENE_INDEX.get(gene)
                        if idx is not None:
                            tfbs_counts[idx] += 1
            tfbs_counts_per_tss.append(tfbs_counts)
        tfbs_df = pd.DataFrame(tfbs_counts_per_tss, columns=config.TF_GENES_K562)
        merged_df = pd.concat([tss_df.reset_index(drop=True), tfbs_df], axis=1)
        all_tss_with_tfbs.append(merged_df)
    result_df = pd.concat(all_tss_with_tfbs, ignore_index=True)
    result_df.to_csv(config.RESULTS_DIR / 'herpesvirus_tss_kinetics_tfbs_counts.csv', index=False)

    # TFBS counts in CDS vs. non-CDS
    cds_df = pd.read_csv(config.PROCESSED_DIR / 'virus_cds_positions.csv')
    motif_with_cds = mark_cres_whose_center_falls_in_regions(
        motif_map_df, cds_df, overlap_col_name='is_cds'
    )
    motif_cds_counts = (
        motif_with_cds.value_counts(['tf_gene', 'is_cds'])
        .reset_index(name='count')
    )
    tfbs_counts = np.zeros((len(config.TF_GENES_K562), 2), dtype=np.float32)
    for tf_gene, is_cds, count in motif_cds_counts.itertuples(index=False):
        for gene in tf_gene.split('-'):
            idx = TF_GENE_INDEX.get(gene)
            if idx is not None:
                tfbs_counts[idx, int(is_cds)] += count
    # calculate total CDS size and genome size (merge overlapping regions)
    cds_bp = 0
    for genome, genome_cds in cds_df.groupby('genome'):
        cds_regions = np.sort(genome_cds[['begin', 'end']].values, axis=1)
        merged = clustering.merge(cds_regions)
        cds_bp += (merged[:, 1] - merged[:, 0]).sum()
    genome_size_bp = pd.read_csv(config.PROCESSED_DIR / 'summary_virus_genome_records.csv').genome_size.sum()
    non_cds_bp = genome_size_bp - cds_bp
    # normalize counts to motifs per 1 kbp
    results_df = pd.DataFrame({
        'tf_gene': config.TF_GENES_K562,
        'not_cds_counts_per_kbp': tfbs_counts[:, 0] * 1000 / non_cds_bp,
        'cds_counts_per_kbp': tfbs_counts[:, 1] * 1000 / cds_bp,
    })
    results_df.to_csv(config.RESULTS_DIR / 'malinois_K562_number_tfbs_per_cds_kbp.csv', index=False)


if __name__ == "__main__":
    main()
