import sys
import numpy as np
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

    for tile_idx, cs_arr in tqdm(
        enumerate(contrib_scores), total=num_tiles, desc='jaccard similarity'
    ):
        cjs_arr[tile_idx] = motifs.batch_rolling_cjs(cs_arr, tf_motifs, cwm_shifts)

    with h5py.File(
        config.PROCESSED_DIR / 'malinois_K562_contribution_scores_to_motifs_similarity', "w"
    ) as f:
        dset = f.create_dataset("continuous_jaccard_similarity", data=cjs_arr, dtype='f')
        dset.attrs["tile_ids"] = tile_ids.astype('S')
        dset.attrs["motif_ids"] = np.array(sorted(tf_motifs.keys()), dtype='S')


def match_motifs_to_contribution_score_peaks(
    tf_motifs: dict[str, np.ndarray],
    cwm_shifts_unordered: dict[str, int],
    contrib_scores: np.ndarray,
    tile_ids: np.ndarray
) -> None:
    tacs_window = 5
    per_pos_threshold = 0.15
    peak_radius = 5
    overlap_threshold = 0.2

    cs_peaks = []
    for tile_idx in tqdm(range(contrib_scores.shape[0]), desc='tiles peak calling'):
        cs_arr = contrib_scores[tile_idx]
        tacs = motifs.rolling_absolute_contribution_scores(cs_arr, window=tacs_window)
        peaks = motifs.call_absolute_contribution_peaks(
            tacs=tacs,
            summit_threshold=per_pos_threshold * tacs_window,
            peak_radius=peak_radius,
            overlap_threshold=overlap_threshold
        )
        matched_tacs = tacs[peaks.mean(1).astype(int)]
        cs_peaks.append((peaks, matched_tacs))

    cjs_path = config.PROCESSED_DIR / 'malinois_K562_contribution_scores_to_motifs_similarity'
    with h5py.File(cjs_path, "r") as f:
        cjs_arr = f["continuous_jaccard_similarity"][()]
        tile_ids = f["continuous_jaccard_similarity"].attrs["tile_ids"].astype(str)
        motif_ids = f["continuous_jaccard_similarity"].attrs["motif_ids"].astype(str)

    cwm_shifts = np.array([cwm_shifts_unordered[motif_id] for motif_id in motif_ids])

    best_match = {}
    for tile_idx, (peak_positions, match_tacs) in tqdm(
        enumerate(cs_peaks), total=len(cs_peaks), desc='map top-1 CJS match'
    ):
        if not peak_positions.size:
            best_match[tile_idx] = {
                "tile_id": tile_ids[tile_idx],
                "positions": [],
                "motifs": [],
                "tacs": [],
                "cjs": []
            }
            continue

        match_pos, match_cjs, match_motif = motifs.find_best_motif_matches(
            cjs_arr[tile_idx],
            motif_ids,
            cwm_shifts,
            peak_positions.mean(1).astype(np.uint8),
            peak_radius,
            top_k=1
        )

        match_positions = [
            [pos, pos + tf_motifs[motif].shape[1]]
            for pos, motif in zip(match_pos.squeeze(1).tolist(), match_motif.squeeze(1).tolist())
        ]

        best_match[tile_idx] = {
            "tile_id": tile_ids[tile_idx],
            "positions": match_positions,
            "motifs": match_motif.squeeze(1).tolist(),
            "tacs": match_tacs.tolist(),
            "cjs": match_cjs.squeeze(1).tolist()
        }

    save_path = config.RESULTS_DIR / "malinois_K562_tf_motif_map.json"
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(best_match, f, ensure_ascii=False, indent=1)


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
