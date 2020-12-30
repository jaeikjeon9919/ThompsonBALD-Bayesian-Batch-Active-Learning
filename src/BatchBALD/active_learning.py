"""
source: https://github.com/BlackHC/batchbald_redux
"""

from typing import Dict, List
import numpy as np
import torch.utils.data as data
import torch
import collections




class ActiveLearningData:
    """Splits `dataset` into an active dataset and an available dataset."""
    dataset: data.Dataset
    training_dataset: data.Dataset
    pool_dataset: data.Dataset
    training_mask: np.ndarray
    pool_mask: np.ndarray

    def __init__(self, dataset: data.Dataset):
        super().__init__()
        self.dataset = dataset
        self.training_mask = np.full((len(dataset), ), False)
        self.validation_mask = np.full((len(dataset), ), False)
        self.pool_mask = np.full((len(dataset), ), True)

        self.training_dataset = data.Subset(self.dataset, None)
        self.validation_dataset = data.Subset(self.dataset, None)
        self.pool_dataset = data.Subset(self.dataset, None)

        self._update_indices()




    def _update_indices(self):
        self.training_dataset.indices = np.nonzero(self.training_mask)[0]
        self.pool_dataset.indices = np.nonzero(self.pool_mask)[0]

    def get_dataset_indices(self, pool_indices: List[int]) -> List[int]:
        """Transform indices (in `pool_dataset`) to indices in the original `dataset`."""
        indices = self.pool_dataset.indices[pool_indices]
        return indices

    def acquire(self, pool_indices):
        """Acquire elements from the pool dataset into the training dataset.

        Add them to training dataset & remove them from the pool dataset."""
        indices = self.get_dataset_indices(pool_indices)

        self.training_mask[indices] = True
        self.pool_mask[indices] = False
        self._update_indices()



    def remove_from_pool(self, pool_indices):
        indices = self.get_dataset_indices(pool_indices)

        self.pool_mask[indices] = False
        self._update_indices()

    def get_random_pool_indices(self, size) -> torch.LongTensor:
        assert 0 <= size <= len(self.pool_dataset)
        pool_indices = torch.randperm(len(self.pool_dataset))[:size]
        return pool_indices

    def extract_dataset_from_pool(self, size) -> data.Dataset:
        """Extract a dataset randomly from the pool dataset and make those indices unavailable.

        Useful for extracting a validation set."""
        return self.extract_dataset_from_pool_from_indices(
            self.get_random_pool_indices(size))

    def extract_dataset_from_pool_from_indices(self, pool_indices) -> data.Dataset:
        """Extract a dataset from the pool dataset and make those indices unavailable.

        Useful for extracting a validation set."""
        dataset_indices = self.get_dataset_indices(pool_indices)

        self.remove_from_pool(pool_indices)
        return data.Subset(self.dataset, dataset_indices)


def get_balanced_sample_indices(target_classes: List, num_classes, n_per_digit=2) -> List[int]:
    """Given `target_classes` randomly sample `n_per_digit` for each of the `num_classes` classes."""
    permed_indices = torch.randperm(len(target_classes))

    if n_per_digit == 0:
        return []

    num_samples_by_class = collections.defaultdict(int)
    initial_samples = []

    for i in range(len(permed_indices)):
        permed_index = int(permed_indices[i])
        index, target = permed_index, int(target_classes[permed_index])

        num_target_samples = num_samples_by_class[target]
        if num_target_samples == n_per_digit:
            continue

        initial_samples.append(index)
        num_samples_by_class[target] += 1

        if len(initial_samples) == num_classes * n_per_digit:
            break

    return initial_samples

def get_subset_base_indices(dataset: data.Subset, indices: List[int]):
    return [int(dataset.indices[index]) for index in indices]


def get_base_indices(dataset: data.Dataset, indices: List[int]):
    if isinstance(dataset, data.Subset):
        return get_base_indices(dataset.dataset, get_subset_base_indices(dataset, indices))
    return indices


class RandomFixedLengthSampler(data.Sampler):
    """
    Sometimes, you really want to do more with little data without increasing the number of epochs.

    This sampler takes a `dataset` and draws `target_length` samples from it (with repetition).
    """
    dataset: data.Dataset
    target_length: int

    def __init__(self, dataset: data.Dataset, target_length: int):
        super().__init__(dataset)
        self.dataset = dataset
        self.target_length = target_length

    def __iter__(self):
        # Ensure that we don't lose data by accident.
        if self.target_length < len(self.dataset):
            return iter(range(len(self.dataset)))

        return iter((torch.randperm(self.target_length) % len(self.dataset)).tolist())

    def __len__(self):
        return self.target_length

