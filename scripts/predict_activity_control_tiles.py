import sys
import pandas as pd

sys.path.insert(0, '..')
from dsdna_mpra import config, boda2  # noqa E402

BATCH_SIZE = 500


def main() -> None:
    # load control tile sequences
    control_df = pd.read_csv(config.RAW_DIR / "control_tile_sequences.csv")

    # load Malinois model
    boda2.unpack_model_artifact(
        config.RAW_DIR / 'malinois_artifacts__20211113_021200__287348.tar.gz',
        config.MALINOIS_MODEL_DIR
    )
    malinois_model = boda2.load_malinois_model(
        model_path=config.MALINOIS_MODEL_DIR
    )

    # Malinois model predictions
    predictions = boda2.compute_malinois_model_predictions(
        malinois_model, control_df.sequence, batch_size=BATCH_SIZE, use_tqdm=True
    )
    predictions_df = pd.DataFrame(
        predictions, columns=['malinois_k562_lfc', 'malinois_hepg2_lfc', 'malinois_sknsh_lfc']
    )
    control_df = pd.concat([control_df, predictions_df], axis=1)
    control_df.to_csv(config.PROCESSED_DIR / "control_tile_malinois_predictions.csv", index=False)


if __name__ == "__main__":
    main()
