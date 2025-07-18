import typing as tp

import torch
from torch import nn
import torch.nn.functional as F
import torchmetrics
import lightning as L

from .. import boda2


def _set_bn_eval(module):
    if isinstance(module, nn.BatchNorm1d):
        module.eval()


class CREClassifier(L.LightningModule):
    def __init__(
        self,
        malinois_model: boda2.BassetBranched,
        num_classes: int,
        internal_features: int = 10,
        regressor_weight: float = .0,
        features_criterion: (tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None) = None,
        regr_loss_function: tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.huber_loss,
        class_loss_function: tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.cross_entropy
    ):
        super().__init__()
        self.num_classes = num_classes
        self.internal_features = internal_features
        self.regressor_weight = regressor_weight

        self.malinois_layers = self._malinois_backbone(malinois_model)
        self.malinois_layers.apply(_set_bn_eval)

        self.internal_fc = nn.Sequential(*[
            nn.Linear(in_features=2600, out_features=self.internal_features),
        ])

        self.regressor = nn.Sequential(*[
            nn.ReLU(), nn.Dropout(p=.67),
            nn.Linear(in_features=self.internal_features, out_features=1)
        ])

        self.classifier = nn.Sequential(*[
            nn.ReLU(), nn.Dropout(p=.67),
            nn.Linear(in_features=self.internal_features, out_features=self.num_classes),  # nn.Softmax(dim=1)
        ])

        self.internal_loss_fn = features_criterion
        self.regr_loss_fn = regr_loss_function
        self.class_loss_fn = class_loss_function

        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=max(num_classes, 2), average="micro"
        )
        self.per_class_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=max(num_classes, 2), average="none"
        )

    @staticmethod
    def _malinois_backbone(model: boda2.BassetBranched) -> list[nn.Module]:
        outer_layers = [
            # encoder layers
            model.pad1, model.conv1, model.nonlin,  # 4 x 600 -> 300 x 600
            model.maxpool_3,  # -> 300 x 200
            model.pad2, model.conv2, model.nonlin,  # -> 200 x 200
            model.maxpool_4,  # -> 200 x 50
        ]
        inner_layers = [
            model.pad3, model.conv3, model.nonlin,  # -> 200 x 50
            model.pad4, model.maxpool_4,  # -> 200 x 13  # TODO [optional]: add convolutions reducing the number of n-channels  # noqa: E501
            nn.Flatten(start_dim=1),
            nn.BatchNorm1d(2600), nn.Dropout(p=.67),
        ]

        for layer in outer_layers:
            for param in layer.parameters():
                param.requires_grad = False

        for layer in inner_layers:
            for param in layer.parameters():
                param.requires_grad = True

        malinois_layers = outer_layers + inner_layers
        return nn.Sequential(*malinois_layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        internal_representation = self.internal_fc(self.malinois_layers(x))
        regression = self.regressor(internal_representation)
        classification = self.classifier(internal_representation)
        return internal_representation, regression, classification

    def predict(self, x: torch.FloatTensor) -> torch.FloatTensor:
        with torch.no_grad():
            _, pred_distances, pred_classes = self.forward(x)
        return pred_distances, pred_classes

    def configure_optimizers(self):
        model_blocks = self.malinois_layers + self.internal_fc + self.classifier + self.regressor
        optimizer = torch.optim.AdamW(model_blocks.parameters(), lr=1e-4, weight_decay=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=1000)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler_config]

    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch, *args):
        return self._step(batch, "valid")

    def _step(self, batch, mode):
        seqs, gt_distances, gt_classes = batch
        internal, pred_distances, pred_classes = self.forward(seqs)
        # shuffled sequence classes do not have assigned distance-to-TSS
        distance_mask = ~(gt_distances.isnan())
        if torch.any(distance_mask):
            loss_regr = self.regr_loss_fn(pred_distances[distance_mask].squeeze(-1), gt_distances[distance_mask])
        else:
            loss_regr = torch.tensor(0.0, device=self.device)
        loss_class = self.class_loss_fn(pred_classes, gt_classes)
        loss = loss_regr * self.regressor_weight + loss_class * (1 - self.regressor_weight)
        if self.internal_loss_fn is not None and mode == "train":
            internal_loss = self.internal_loss_fn(internal, gt_classes)
            loss = .9 * loss + .1 * internal_loss
        accs = self.accuracy(pred_classes.argmax(axis=-1), gt_classes)
        per_class_accs = self.per_class_accuracy(pred_classes.argmax(axis=-1), gt_classes)
        metrics = {
            f"{mode}_accs": accs,
            f"{mode}_classification_loss": loss_class,
            f"{mode}_regression_loss": loss_regr,
            f"{mode}_total_loss": loss,
        }
        for cl_idx, cl_accs in enumerate(per_class_accs):
            if cl_idx in gt_classes:
                metrics[f"{mode}_accs_{cl_idx}"] = cl_accs
        self.log_dict(
            metrics,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss
