import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import h5py

sys.path.insert(0, '..')
from dsdna_mpra import config, motifs  # noqa E402


def compute_jaccard_similarity_scores(
    tf_motifs: dict[str, np.ndarray],
    cwm_shifts: dict[str, int],
    contrib_scores: np.ndarray,
    tile_ids: np.ndarray
) -> None:

    num_tiles, num_motifs = contrib_scores.shape[0], len(tf_motifs)
    cjs_arr = np.zeros((num_tiles, num_motifs, 200))
    with tqdm(total=num_tiles, desc='jaccard similarity', dynamic_ncols=True, leave=False) as pbar:
        for tile_idx, cs_arr in enumerate(contrib_scores):
            cjs_arr[tile_idx] = motifs.batch_rolling_cjs(cs_arr, tf_motifs, cwm_shifts)

            if (tile_idx + 1) % 10000 == 0 or tile_idx + 1 == num_tiles:
                step = 10000 if tile_idx + 1 < num_tiles else num_tiles % 10000
                pbar.update(step)

    with h5py.File(
        config.PROCESSED_DIR / 'malinois_K562_contribution_scores_to_motifs_similarity', "w"
    ) as f:
        _ = f.create_dataset("continuous_jaccard_similarity", data=cjs_arr, dtype='f')
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset("tile_ids", data=tile_ids.astype(dt))
        f.create_dataset("motif_ids", data=np.array(sorted(tf_motifs.keys()), dtype=dt))


def match_motifs_to_contribution_score_peaks(
    tf_motifs: dict[str, np.ndarray],
    cwm_shifts_unordered: dict[str, int],
    contrib_scores: np.ndarray,
    tile_ids: np.ndarray
) -> None:

    cs_peaks = []
    num_tiles = contrib_scores.shape[0]
    with tqdm(total=num_tiles, desc='tiles peak calling', dynamic_ncols=True, leave=False) as pbar:
        for tile_idx in range(num_tiles):
            cs_arr = contrib_scores[tile_idx]
            tacs = motifs.rolling_absolute_contribution_scores(cs_arr, window=config.TACS_WINDOW)
            peaks = motifs.call_absolute_contribution_peaks(
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

    cjs_path = config.PROCESSED_DIR / 'malinois_K562_contribution_scores_to_motifs_similarity'
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

            match_pos, match_cjs, match_motif = motifs.find_best_motif_matches(
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

    save_path = config.RESULTS_DIR / "malinois_K562_tf_motif_map.json"
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(best_match, f, ensure_ascii=False, indent=1)


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

    compute_jaccard_similarity_scores(tf_motifs, cwm_shifts, contrib_scores, tile_ids)
    match_motifs_to_contribution_score_peaks(tf_motifs, cwm_shifts, contrib_scores, tile_ids)
    build_genome_wide_motif_instances_map()
