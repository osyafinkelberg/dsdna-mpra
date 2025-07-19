import sys
import numpy as np
import pandas as pd
import json

sys.path.insert(0, '..')
from dsdna_mpra import config, motifs  # noqa E402


def build_genome_wide_motif_instances_map() -> None:
    # load list of virus genomes
    virus_genomes = pd.read_csv(config.RAW_DIR / 'virus_genbank_ids.txt').columns.values
    # load and filter paired tiles
    paired_tiles = pd.read_csv(config.PROCESSED_DIR / 'virus_paired_tiles_log2p_ratios.csv')
    paired_tiles = paired_tiles[
        paired_tiles['genome'].isin(virus_genomes) &
        paired_tiles['family'].isin(config.DSDNA_FAMILIES)
    ]
    paired_tiles = (
        paired_tiles
        .drop_duplicates(['genome', 'begin'])[
            ['family', 'strain', 'tile_id', 'genome', 'begin']
        ]
        .sort_values(['genome', 'begin'])
        .reset_index(drop=True)
    )
    # load tile-to-motif mapping
    with open(config.RESULTS_DIR / "malinois_K562_tf_motif_map.json", 'r', encoding='utf-8') as f:
        tile_motif_map = {
            tile_map['tile_id']: tile_map
            for tile_map in json.load(f)
        }
    # load motif-to-gene mapping
    motif_to_gene_map = dict(pd.read_csv(config.RESULTS_DIR / 'malinois_K562_tf_motif_families.csv').values)

    # build genome-wide motif annotation
    def pick_max_tacs_match(group: pd.DataFrame, include_groups: bool = False) -> pd.Series:
        """
        Selects the row with the highest contribution score from overlapping motif instances.
        """
        return group.loc[group['contribution_score'].idxmax()]

    all_genome_maps = []
    for genome_id, genome_tiles in paired_tiles.groupby('genome'):
        motif_annotations = []
        for family, strain, tile_id, genome, tile_start in genome_tiles.itertuples(index=False):
            tile_data = tile_motif_map[tile_id]
            motifs = tile_data['motifs']
            positions = tile_data['motif_positions']
            tacs_scores = tile_data['tacs']
            cj_scores = tile_data['cjs']
            for motif, (start, end), tacs, cjs in zip(motifs, positions, tacs_scores, cj_scores):
                motif_clean = motif.removesuffix('_fwd').removesuffix('_rev')
                gene = motif_to_gene_map.get(motif_clean, 'Unknown')
                motif_annotations.append({
                    'family': family,
                    'strain': strain,
                    'tile_id': tile_id,
                    'genome': genome,
                    'tile_begin': tile_start,
                    'tf_motif': motif,
                    'tf_gene': gene,
                    'begin': tile_start + start,
                    'end': tile_start + end,
                    'contribution_score': tacs,
                    'continuous_jaccard_similarity': cjs
                })
        motif_df = pd.DataFrame(motif_annotations)
        # deduplicate overlapping motif instances using a unique identifier
        motif_df['instance_id'] = motif_df[['tf_motif', 'begin', 'end']].astype(str).agg('-'.join, axis=1)
        motif_df = motif_df.groupby('instance_id').apply(pick_max_tacs_match, include_groups=False)
        motif_df.sort_values(['begin', 'end', 'tile_begin'], inplace=True)
        all_genome_maps.append(motif_df.reset_index(drop=True))

    # concatenate all genome-level maps and save
    final_df = pd.concat(all_genome_maps).reset_index(drop=True)
    output_path = config.RESULTS_DIR / 'malinois_K562_tf_motif_genome_annotation.csv'
    final_df.to_csv(output_path, index=False)


if __name__ == "__main__":

    tf_motifs = motifs.parse_pwm_file(config.RESULTS_DIR / 'malinois_K562_tf_motifs.cb')
    cwm_shifts = {
        motif_id: motifs.rolling_absolute_contribution_scores(cwm, window=5).argmax()
        for motif_id, cwm in tf_motifs.items()
    }

    contrib_scores_data = np.load(
        config.PROCESSED_DIR / "malinois_K562_contribution_scores.npz", allow_pickle=True
    )
    contrib_scores = contrib_scores_data['arr_0']
    tile_ids = np.concatenate([
        contrib_scores_data['virus_tile_ids'],
        contrib_scores_data['dhs_tile_ids']
    ])

    cjs_output_path = config.PROCESSED_DIR / 'malinois_K562_contribution_scores_to_motifs_similarity'
    motifs.compute_jaccard_similarity_scores(tf_motifs, cwm_shifts, contrib_scores, tile_ids, cjs_output_path)

    motif_output_path = config.RESULTS_DIR / "malinois_K562_tf_motif_map.json"
    motifs.match_motifs_to_contribution_score_peaks(tf_motifs, cwm_shifts, contrib_scores, tile_ids, motif_output_path)

    build_genome_wide_motif_instances_map()
