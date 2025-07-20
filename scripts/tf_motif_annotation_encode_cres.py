import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, '..')
from dsdna_mpra import config, boda2, motifs  # noqa E402

BATCH_SIZE = 500


def compute_contribution_scores() -> None:
    # load ENCODE CREs data (raw data preprocessed in the `cre_classifier_dataset_preparation.py`)
    encode_df = pd.read_csv(config.PROCESSED_DIR / 'encode/cre_classifier_dataset.csv')
    cre_sequences = encode_df.sequence.str.slice(50, 250).values  # central 200 bp sequence
    cre_ids = encode_df.encode_id.values

    # onehot-encoded sequences
    valid_mask, onehots = boda2.convert_sequences_to_malinois_input(cre_sequences)
    assert all(valid_mask)
    onehots = onehots.cpu()
    np.savez(
        config.PROCESSED_DIR / "encode/cre_classifier_dataset_onehot_sequences.npz",
        cre_ids=cre_ids, arr_0=onehots[..., 200: 400]
    )

    # load Malinois model
    boda2.unpack_model_artifact(
        config.RAW_DIR / 'malinois_artifacts__20211113_021200__287348.tar.gz',
        config.MALINOIS_MODEL_DIR
    )
    malinois_model = boda2.load_malinois_model(
        model_path=config.MALINOIS_MODEL_DIR
    )

    # hypothetical contribution scores (`pred_idx=0` corresponds to K562 cell line in Malinois model outputs)
    cre_scores = boda2.compute_model_contribution_scores(
        malinois_model, cre_sequences, pred_idx=0, batch_size=BATCH_SIZE, use_tqdm=True
    )
    np.savez(
        config.PROCESSED_DIR / "encode/cre_classifier_dataset_malinois_k562_contribution_scores.npz",
        cre_ids=cre_ids, arr_0=cre_scores
    )


def tf_motif_annotation() -> None:
    # load Malinois K562 TF motif data (outputs of the `tfmodisco_postprocessing.py` script)
    tf_motifs = motifs.parse_pwm_file(config.RESULTS_DIR / 'malinois_K562_tf_motifs.cb')
    cwm_shifts = {
        motif_id: motifs.rolling_absolute_contribution_scores(cwm, window=5).argmax()
        for motif_id, cwm in tf_motifs.items()
    }

    # contribution scores precomputed in this script
    contrib_scores_data = np.load(
        config.PROCESSED_DIR / "encode/cre_classifier_dataset_malinois_k562_contribution_scores.npz",
        allow_pickle=True
    )
    contrib_scores = contrib_scores_data['arr_0']
    cre_ids = contrib_scores_data['cre_ids']

    cjs_output_path = config.PROCESSED_DIR / 'encode/cre_classifier_dataset_malinois_k562_contribution_scores_motifs'
    motifs.compute_jaccard_similarity_scores(tf_motifs, cwm_shifts, contrib_scores, cre_ids, cjs_output_path)

    motif_output_path = config.RESULTS_DIR / "encode_cres_malinois_K562_tf_motif_map.json"
    motifs.match_motifs_to_contribution_score_peaks(
        tf_motifs, cwm_shifts, contrib_scores, cjs_output_path, motif_output_path
    )


def tf_motif_statistics() -> None:
    encode_df = pd.read_csv(config.PROCESSED_DIR / 'encode/cre_classifier_dataset.csv')
    motif_to_gene_map = dict(pd.read_csv(config.RESULTS_DIR / 'malinois_K562_tf_motif_families.csv').values)
    with open(config.RESULTS_DIR / "encode_cres_malinois_K562_tf_motif_map.json", 'r', encoding='utf-8') as f:
        cre_motif_map = {entry['tile_id']: entry for entry in json.load(f)}

    cre_columns = ['encode_id', 'chromosome', 'begin', 'end', 'encode_region_type', 'malinois_k562_lfc']
    motifs.compute_tfbs_counts(
        input_df=encode_df,
        id_func=lambda row: row.encode_id,
        output_path=config.RESULTS_DIR / "malinois_K562_tfbs_counts_encode_cres.csv",
        base_columns=cre_columns,
        tile_motif_map=cre_motif_map,
        motif_to_gene_map=motif_to_gene_map
    )


def main() -> None:
    compute_contribution_scores()
    tf_motif_annotation()
    tf_motif_statistics()


if __name__ == "__main__":
    main()
