import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '..')
from dsdna_mpra import config, boda2  # noqa E402

BATCH_SIZE = 500


def main() -> None:
    # load virus tile and human DHS data
    paired_tiles = pd.read_csv(config.PROCESSED_DIR / 'virus_paired_tiles_sequences.csv')
    tile_sequences = paired_tiles.tile_sequence.values
    tile_ids = paired_tiles.tile_id.values
    dhs_df = pd.read_csv(config.RAW_DIR / 'K562-DS15363_peaks_with_sequences.bed', sep=',')
    assert all(dhs_df.sequence.str.len() == 200)  # central 200 bp sequence
    dhs_sequences = dhs_df.sequence.values
    dhs_ids = (dhs_df.chromosome + '-' + dhs_df.center.astype(str)).values

    # onehot-encoded sequences
    valid_mask, onehots = boda2.convert_sequences_to_malinois_input(
        np.concatenate([tile_sequences, dhs_sequences])
    )
    assert all(valid_mask)
    onehots = onehots.cpu()
    np.savez(
        config.PROCESSED_DIR / "malinois_K562_onehot_sequences.npz",
        virus_tile_ids=tile_ids, virus_tiles=onehots[:tile_ids.size, :, 200: 400],
        dhs_tile_ids=dhs_ids, dhs_tiles=onehots[tile_ids.size:, :, 200: 400],
        arr_0=onehots[:, :, 200: 400]
    )

    # load Malinois model
    boda2.unpack_model_artifact(
        config.RAW_DIR / 'malinois_artifacts__20211113_021200__287348.tar.gz',
        config.MALINOIS_MODEL_DIR
    )
    malinois_model = boda2.load_malinois_model(
        model_path=config.MALINOIS_MODEL_DIR
    )

    # Malinois model predictions
    tile_activities = boda2.compute_malinois_model_predictions(
        malinois_model, tile_sequences, batch_size=BATCH_SIZE, use_tqdm=True
    )
    dhs_activities = boda2.compute_malinois_model_predictions(
        malinois_model, dhs_sequences, batch_size=BATCH_SIZE, use_tqdm=True
    )
    np.savez(
        config.RESULTS_DIR / "malinois_predicted_activities.npz",
        virus_tile_ids=tile_ids, virus_tiles=tile_activities,
        dhs_tile_ids=dhs_ids, dhs_tiles=dhs_activities,
    )

    # hypothetical contribution scores (`pred_idx=0` corresponds to K562 cell line in Malinois model outputs)
    tile_scores = boda2.compute_model_contribution_scores(
        malinois_model, tile_sequences, pred_idx=0, batch_size=BATCH_SIZE, use_tqdm=True
    )
    dhs_scores = boda2.compute_model_contribution_scores(
        malinois_model, dhs_sequences, pred_idx=0, batch_size=BATCH_SIZE, use_tqdm=True
    )
    np.savez(
        config.PROCESSED_DIR / "malinois_K562_contribution_scores.npz",
        virus_tile_ids=tile_ids, virus_tiles=tile_scores,
        dhs_tile_ids=dhs_ids, dhs_tiles=dhs_scores,
        arr_0=np.concatenate([tile_scores, dhs_scores])
    )


if __name__ == "__main__":
    main()
