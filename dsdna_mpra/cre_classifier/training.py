from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from . import dataloader
from . import model


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.0):
        assert 0 <= smoothing < 1
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, p, y):
        with torch.no_grad():
            K = self.num_classes
            y_one_hot = F.one_hot(y, num_classes=K)
            smooth_y = (1 - self.smoothing) * y_one_hot + (self.smoothing / K)
        loss = F.kl_div(F.log_softmax(p, dim=-1), smooth_y, reduction='batchmean')
        return loss


def train_cre_classifier_model(
    model: model.CREClassifier,
    train_data_path: Path,
    test_data_path: Path,
    log_dir_path: Path
):
    NUM_EPOCHS = 125
    BATCH_SIZE = (2048 // model.num_classes) * model.num_classes  # must be a multiplier of num_classes!  # 1000
    NUM_WORKERS = 2
    CALLBACKS = [
        L.pytorch.callbacks.TQDMProgressBar(),
        L.pytorch.callbacks.LearningRateMonitor(),
        L.pytorch.callbacks.ModelCheckpoint(
            filename="{epoch}-{valid_accs:.3f}",
            monitor="valid_accs",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        L.pytorch.callbacks.EarlyStopping(
            monitor="valid_accs",
            mode="max",
            patience=1000,  # for experimental purposes
            verbose=False,
        )
    ]

    # dl_train = torch.utils.data.DataLoader(
    #     custom_dataset.CREDataset(mode="train", data_path=train_data_path),
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     drop_last=True,
    #     num_workers=NUM_WORKERS,
    #     persistent_workers=True,
    # )

    ds_train = dataloader.CREDataset(
        mode="train", data_path=train_data_path
    )
    dl_train = torch.utils.data.DataLoader(
        ds_train,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        sampler=dataloader.ClassBalancedIndexSampler(
            cre_dataset=ds_train,
            batch_size=BATCH_SIZE,
            n_passes=100,
        ),
    )

    dl_valid = torch.utils.data.DataLoader(
        dataloader.CREDataset(
            mode="test", data_path=test_data_path
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
    )

    tb_logger = TensorBoardLogger(
        save_dir=log_dir_path,
        name=""
    )

    trainer = L.Trainer(
        accelerator="auto",
        max_epochs=NUM_EPOCHS,
        callbacks=CALLBACKS,
        default_root_dir=log_dir_path,
        log_every_n_steps=10,
        logger=tb_logger,
        enable_checkpointing=True,
        strategy='ddp_notebook',
    )

    trainer.fit(model, dl_train, dl_valid)
