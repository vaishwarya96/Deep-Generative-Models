import torch
from torch.utils import data
from torch.utils.data import Dataset

from torchvision import datasets, transforms

def get_MNIST(root="./"):

    input_size = 64
    num_channels = 1
    transform_list = [transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize((0.2861,), (0.3530,))]
    transform = transforms.Compose(transform_list)

    train_dataset = datasets.MNIST(
        root + "data/MNIST", train=True, download=True, transform=transform
    )

    test_dataset = datasets.MNIST(
        root + "data/MNIST", train=False, download=True, transform=transform
    )

    return train_dataset, num_channels

def get_FashionMNIST(root="./"):
    input_size = 64
    num_channels = 1

    transform_list = [transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize((0.2861,), (0.3530,))]
    transform = transforms.Compose(transform_list)

    train_dataset = datasets.FashionMNIST(
        root + "data/", train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root + "data/", train=False, download=True, transform=transform
    )
    return train_dataset, num_channels

def get_SVHN(root="./"):
    input_size = 64
    num_channels = 3
    transform = transforms.Compose(
        [transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = datasets.SVHN(
        root + "data/SVHN", split="train", transform=transform, download=True
    )
    test_dataset = datasets.SVHN(
        root + "data/SVHN", split="test", transform=transform, download=True
    )
    return train_dataset, num_channels

def get_CIFAR10(root="./"):
    input_size = 64
    num_channels = 3
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=True, transform=transform, download=True
    )

    test_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=False, transform=transform, download=True
    )

    return train_dataset, num_channels

def get_celeba(root="./"):
    input_size = 64
    num_channels = 3
    transform=transforms.Compose([
                               transforms.Resize(input_size),
                               transforms.CenterCrop(input_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ])

    dataset = datasets.ImageFolder(root="./data/celeba", transform=transform)

    return dataset, num_channels

all_datasets = {
    "MNIST": get_MNIST,
    "FashionMNIST": get_FashionMNIST,
    "SVHN": get_SVHN,
    "CIFAR10": get_CIFAR10,
    "Celeba": get_celeba,
}

def get_dataloaders(args, root="./"):
    ds = all_datasets[args.dataset](root)
    dataset, num_channels = ds

    kwargs = {"num_workers": args.num_workers, "pin_memory": True}

    train_loader = data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, **kwargs
    )

    return train_loader, num_channels
