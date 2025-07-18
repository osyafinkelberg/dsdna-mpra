from pathlib import Path

import numpy as np
import torch

from collections import defaultdict


def shift_augmentation(  # optional: has minimal effect on overall results
    flanked_oh: torch.Tensor,
    extended_oh: torch.Tensor,
    shift: int
) -> torch.Tensor:
    extend_sz = extended_oh.shape[1]
    assert abs(shift) <= extend_sz / 2 - 100  # tile size 200 bp
    flanked_oh[:, 200:400] = extended_oh[:, extend_sz // 2 - 100 - shift: extend_sz // 2 + 100 - shift]
    return flanked_oh


class CREDataset(torch.utils.data.Dataset):
    def __init__(self, mode: str, data_path: Path, random_seed: int = 42):
        super().__init__()
        assert mode in ["train", "test"]  # no augmentations for "test"
        self.mode = mode
        self.rng = np.random.default_rng(random_seed)

        data = np.load(data_path, allow_pickle=True)
        self.original_index = data['original_index']
        self.sequences_extended = torch.from_numpy(data['sequence_with_genome_context']).float()
        self.sequences_flanked = torch.from_numpy(data['sequence_with_mpra_flanks']).float()
        self.classes = torch.from_numpy(data['class']).long()
        self.tss_distances = torch.from_numpy(data['tss_distance_log10p']).float()
        self.tss_strands = torch.from_numpy(data['tss_strand']).float()
        self.is_shuffled = data['sequence_is_shuffled']

    def __len__(self) -> int:
        return len(self.sequences_flanked)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence_flanked = self.sequences_flanked[index]
        tss_distance = self.tss_distances[index]
        cre_class = self.classes[index]
        if self.mode == "train" and not self.is_shuffled[index]:
            shift = int(self.rng.integers(low=-5, high=6, size=1)) * 10
            sequence_flanked = shift_augmentation(sequence_flanked, self.sequences_extended[index], shift)
            tss_distance = torch.log10(torch.abs(tss_distance - shift * self.tss_strands[index]) + 1)
        return sequence_flanked, tss_distance, cre_class


class ClassBalancedIndexSampler(torch.utils.data.sampler.Sampler[int]):
    def __init__(self, cre_dataset: CREDataset, batch_size: int, n_passes: int) -> None:
        self.cre_dataset = cre_dataset
        self.class_to_indices = defaultdict(list)
        for index, cre_class in enumerate(cre_dataset.classes):
            class_idx = int(cre_class.item())
            self.class_to_indices[class_idx].append(index)
        self.class_to_indices = dict(self.class_to_indices)
        self.unique_classes = cre_dataset.classes.unique().tolist()
        self.n_classes = len(self.unique_classes)
        self.batch_size = batch_size
        self.elems_per_class_per_batch = self.batch_size // self.n_classes
        self.n_passes = n_passes

    def __iter__(self):
        for _ in range(self.n_passes):
            for class_idx in self.unique_classes:
                class_idxs = self.class_to_indices[class_idx]
                class_batch_idxs = np.random.choice(class_idxs, size=self.elems_per_class_per_batch, replace=True)
                for idx in class_batch_idxs:
                    yield idx

    def __len__(self) -> int:
        return self.batch_size * self.n_passes
