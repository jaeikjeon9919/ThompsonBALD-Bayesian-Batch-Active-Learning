"""
source: https://github.com/BlackHC/batchbald_redux
"""

import PIL
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.utils.data as data
import torch

PX_SIZE = 28

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

mnist_train_transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.Resize((PX_SIZE, PX_SIZE), interpolation=PIL.Image.LANCZOS),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomCrop(PX_SIZE, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307, 0.1307, 0.1307], std=[0.3081, 0.3081, 0.3081])
])

mnist_test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((PX_SIZE, PX_SIZE), interpolation=PIL.Image.LANCZOS),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307, 0.1307, 0.1307], std=[0.3081, 0.3081, 0.3081])
])

class TransformedDataset(data.Dataset):
    """
    Transforms a dataset.

    Arguments:
        dataset (Dataset): The whole Dataset
        transformer (LambdaType): (idx, sample) -> transformed_sample
    """

    def __init__(self, dataset, *, transformer=None, vision_transformer=None):
        self.dataset = dataset
        assert not transformer or not vision_transformer
        if transformer:
            self.transformer = transformer
        else:
            self.transformer = lambda _, data_label: (vision_transformer(data_label[0]), data_label[1])

    def __getitem__(self, idx):
        return self.transformer(idx, self.dataset[idx])

    def __len__(self):
        return len(self.dataset)



def create_repeated_MNIST_dataset(model_type, *, num_repetitions: int = 3,
                                  add_noise: bool = True):
    # num_classes = 10, input_size = 28



    if model_type == "MLP":
        train_dataset = datasets.MNIST(root='./data',
                                       train=True,
                                       download=True,
                                       transform=transform)

    elif model_type == "ResNet":
        train_dataset = datasets.MNIST(root='./data',
                                       train=True,
                                       download=True,
                                       transform=mnist_train_transform)

    if num_repetitions > 1:
        train_dataset = data.ConcatDataset([train_dataset] * num_repetitions)

    if add_noise:
        dataset_noise = torch.empty((len(train_dataset), 28, 28),
                                    dtype=torch.float32).normal_(0.0, 0.1)

        def apply_noise(idx, sample):
            data, target = sample
            return data + dataset_noise[idx], target

        train_dataset = TransformedDataset(train_dataset,
                                           transformer=apply_noise)
    if model_type == "MLP":
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    elif model_type == "ResNet":
        test_dataset = datasets.MNIST(root='./data', train=False, transform=mnist_test_transform, download=True)

    return train_dataset, test_dataset


def create_MNIST_dataset():
    return create_repeated_MNIST_dataset(num_repetitions=1, add_noise=False)


def get_targets(dataset):
    """Get the targets of a dataset without any target transforms.

    This supports subsets and other derivative datasets."""
    if isinstance(dataset, TransformedDataset):
        return get_targets(dataset.dataset)
    if isinstance(dataset, data.Subset):
        targets = get_targets(dataset.dataset)
        return torch.as_tensor(targets)[dataset.indices]
    if isinstance(dataset, data.ConcatDataset):
        return torch.cat([get_targets(sub_dataset) for sub_dataset in dataset.datasets])

    return torch.as_tensor(dataset.targets)


##

