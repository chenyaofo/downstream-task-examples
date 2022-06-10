import os
import torch
import torchvision
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

from conf import config

train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

data_module = CIFAR10DataModule(
    data_dir=config["data_root"],
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms,
)