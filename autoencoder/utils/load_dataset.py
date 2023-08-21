import torch
from torch.utils import data
from torch.utils.data import Dataset

from torchvision import datasets, transforms

def get_MNIST(root="./"):
    input_size = 28
    num_classes = 10
    transform_list = [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.2861,), (0.3530,))]
    transform = transforms.Compose(transform_list)

    train_dataset = datasets.MNIST(
        root + "data/MNIST", train=True, download=True, transform=transform
    )

    test_dataset = datasets.MNIST(
        root + "data/MNIST", train=False, download=True, transform=transform
    )
    return input_size, num_classes, train_dataset, test_dataset


def get_FashionMNIST(root="./"):
    input_size = 28
    num_classes = 10

    transform_list = [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.2861,), (0.3530,))]
    transform = transforms.Compose(transform_list)

    train_dataset = datasets.FashionMNIST(
        root + "data/", train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root + "data/", train=False, download=True, transform=transform
    )
    return input_size, num_classes, train_dataset, test_dataset


def get_SVHN(root="./"):
    input_size = 32
    num_classes = 10
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = datasets.SVHN(
        root + "data/SVHN", split="train", transform=transform, download=True
    )
    test_dataset = datasets.SVHN(
        root + "data/SVHN", split="test", transform=transform, download=True
    )
    return input_size, num_classes, train_dataset, test_dataset


def get_CIFAR10(root="./"):
    input_size = 32
    num_classes = 10
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=False, transform=test_transform, download=True
    )

    return input_size, num_classes, train_dataset, test_dataset

all_datasets = {
    "MNIST": get_MNIST,
    "FashionMNIST": get_FashionMNIST,
    "SVHN": get_SVHN,
    "CIFAR10": get_CIFAR10,
}

def get_dataset(dataset, root="./"):
    return all_datasets[dataset](root)

def get_dataloaders(args, root="./"):
    ds = all_datasets[args.dataset](root)
    input_size, num_classes, train_dataset, test_dataset = ds

    kwargs = {"num_workers": args.num_workers, "pin_memory": True}

    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
    )

    test_loader = data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
    )

    return train_loader, test_loader