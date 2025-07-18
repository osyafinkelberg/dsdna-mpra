import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, '..')
from dsdna_mpra import config  # noqa E402
from dsdna_mpra import boda2, cre_classifier  # noqa E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model() -> torch.nn.Module:
    malinois_model = boda2.load_malinois_model()
    model = cre_classifier.model.CREClassifier(
        malinois_model,
        num_classes=len(config.ENCODE_CRE_TYPES),
        internal_features=100,
    )
    model_path = config.RESULTS_DIR / "cre_classifier/internfeats-100_regrweight-05_huberloss_labelsmoothing-01.pt"
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    return model.eval().to(DEVICE)


def predict(model, dataloader):
    logits_all, targets_all = [], []
    with torch.no_grad():
        for x, _, y in tqdm(dataloader, desc="Test set predictions"):
            _, logits = model.predict(x.to(DEVICE))
            logits_all.extend(logits.tolist())
            targets_all.extend(y.tolist())
    return np.array(logits_all), np.array(targets_all)


def create_prediction_df(logits, true_labels=None):
    class_names = np.array(config.ENCODE_CRE_TYPES)
    pred_df = pd.DataFrame(
        logits,
        columns=[name.lower().replace(' ', '_') + '_logits' for name in class_names]
    )
    pred_df['predicted_class'] = class_names[logits.argmax(axis=1)]
    if true_labels is not None:
        pred_df['real_class'] = class_names[true_labels]
    return pred_df


def main():
    model = load_model()

    # --- ENCODE dataset prediction ---
    test_dataset = cre_classifier.dataloader.CREDataset(
        mode="test",
        data_path=config.PROCESSED_DIR / "encode/cre_classifier_test_dataset.npz",
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1024, shuffle=False, drop_last=False, num_workers=1
    )
    logits, true_labels = predict(model, test_loader)
    encode_df = pd.read_csv(config.PROCESSED_DIR / 'encode/cre_classifier_dataset.csv')
    encode_df = encode_df.iloc[test_dataset.original_index].reset_index(drop=True)

    pred_df = create_prediction_df(logits, true_labels)
    meta_cols = ['chromosome', 'begin', 'end', 'malinois_k562_lfc', 'malinois_hepg2_lfc', 'malinois_sknsh_lfc']
    result_df = pd.concat([encode_df[meta_cols], pred_df], axis=1)
    result_df.to_csv(config.RESULTS_DIR / 'encode_validation_classification.csv', index=False)

    # --- virus tile prediction ---
    virus_genomes = pd.read_csv(config.RAW_DIR / 'virus_genbank_ids.txt').columns.values
    paired_tiles = pd.read_csv(config.PROCESSED_DIR / 'virus_paired_tiles_log2p_ratios.csv')
    paired_tiles = paired_tiles[
        paired_tiles['genome'].isin(virus_genomes) &
        paired_tiles['family'].isin(config.DSDNA_FAMILIES) &
        (paired_tiles.cell == 'K562')
    ].reset_index(drop=True)

    thresholds = pd.read_csv(config.RESULTS_DIR / 'thresholds_log2_1p.csv')
    K562_THRESHOLD = thresholds[thresholds.cell == 'K562'].threshold.iloc[0]
    active_tiles = paired_tiles[
        paired_tiles[['fwd_lfc', 'rev_lfc']].max(axis=1) >= K562_THRESHOLD
    ].reset_index(drop=True)

    seqs = pd.read_csv(config.PROCESSED_DIR / 'virus_paired_tiles_sequences.csv')
    active_tiles = active_tiles.merge(seqs[['tile_id', 'tile_sequence']], on='tile_id', how='left')

    cds = pd.read_csv(config.RESULTS_DIR / "virus_paired_tiles_cds_overlap.csv")
    active_tiles = active_tiles.merge(cds.drop_duplicates('tile_id')[['tile_id', 'is_cds']], on='tile_id', how='left')

    valid_mask, tiles_oh = boda2.convert_sequences_to_malinois_input(active_tiles.tile_sequence)
    assert all(valid_mask)

    tiles_oh_rc = tiles_oh.clone()
    tiles_oh_rc[..., 200:400] = tiles_oh_rc[..., 200:400].flip(dims=[1, 2])

    with torch.no_grad():
        _, logits_fwd = model.predict(tiles_oh.to(DEVICE))
        _, logits_rev = model.predict(tiles_oh_rc.to(DEVICE))

    logits_avg = (logits_fwd + logits_rev).cpu().numpy() / 2
    pred_df = create_prediction_df(logits_avg)

    pred_df['is_cds'] = active_tiles.is_cds
    pred_df.insert(0, 'family', active_tiles.family)
    pred_df.insert(1, 'strain', active_tiles.strain)
    pred_df.insert(2, 'tile_id', active_tiles.tile_id)
    pred_df.to_csv(config.RESULTS_DIR / 'k562_active_tiles_classification.csv', index=False)


if __name__ == "__main__":
    main()
