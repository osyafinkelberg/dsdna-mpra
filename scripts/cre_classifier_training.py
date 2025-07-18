import sys
import torch

sys.path.insert(0, '..')
from dsdna_mpra import config  # noqa E402
from dsdna_mpra import boda2, cre_classifier  # noqa E402


def main() -> None:

    # construct the CRE classifier model with specified parameters
    malinois_model = boda2.load_malinois_model()
    num_classes = len(config.ENCODE_CRE_TYPES)
    model = cre_classifier.model.CREClassifier(
        malinois_model=malinois_model,
        num_classes=num_classes,
        internal_features=100,
        regressor_weight=0.5,
        features_criterion=None,
        regr_loss_function=torch.nn.functional.huber_loss,
        class_loss_function=cre_classifier.training.LabelSmoothingLoss(
            num_classes=num_classes,
            smoothing=0.1
        ),
    )

    # train the model using training and test datasets
    train_data = config.PROCESSED_DIR / "encode/cre_classifier_train_dataset.npz"
    test_data = config.PROCESSED_DIR / "encode/cre_classifier_test_dataset.npz"
    log_dir = config.RESULTS_DIR / "cre_classifier/training_logs/"
    cre_classifier.training.train_cre_classifier_model(
        model=model,
        train_data_path=train_data,
        test_data_path=test_data,
        log_dir_path=log_dir
    )

    # saves the model state dictionary to the specified file
    filename = 'internfeats-100_regrweight-05_huberloss_labelsmoothing-01_replica.pt'
    torch.save(model.state_dict(), config.RESULTS_DIR / f"cre_classifier/{filename}")


if __name__ == "__main__":
    main()
