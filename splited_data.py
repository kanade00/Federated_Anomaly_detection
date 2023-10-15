import os
import numpy as np
from PIL import Image
from skimage.filters import gaussian as gblur
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data import Subset, ConcatDataset
from torch.utils.data import random_split
from copy import deepcopy

# ref: https://github.com/HobbitLong/SupContrast
class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def cifar10(
    data_dir, batch_size=32, mode="base", normalize=True, norm_layer=None, size=32,
        ratio_pollution=0.0, num_clients=10
):
    """
    mode: org | base | ssl
    """
    assert num_clients % 10 == 0

    transform_train = [
        transforms.Resize(size),
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    transform_test = [transforms.Resize(size), transforms.ToTensor()]

    if mode == "org":
        None
    elif mode == "base":
        transform_train = [transforms.Resize(size), transforms.ToTensor()]
    elif mode == "ssl":
        transform_train = [
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]
    else:
        raise ValueError(f"{mode} mode not supported")

    if norm_layer is None:
        norm_layer = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201]
        )
    
    if normalize:
        transform_train.append(norm_layer)
        transform_test.append(norm_layer)

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    if mode == "ssl":
        transform_train = TwoCropTransform(transform_train)

    trainset = MyCIFAR10(
        root=os.path.join(data_dir, "cifar10"),
        train=True,
        download=True,
        transform=transform_train,
    )
    testset = datasets.CIFAR10(
        root=os.path.join(data_dir, "cifar10"),
        train=False,
        download=True,
        transform=transform_test,
    )

    ood_trainset = MyCIFAR100(
        root=os.path.join(data_dir, "cifar100"),
        train=True,
        download=True,
        transform=transform_train,
    )

    shifted_trainset = get_shifted_trainset(trainset, ood_trainset, ratio_pollution)

    splited_trainset = []

    for i in range(10):
        num_split_client = num_clients // 10
        splited_trainset += random_split_dataset(shifted_trainset[i], num_split_client)
    # trainset = get_trainset(trainset, ood_trainset, ratio_pollution)
    # splited_trainset = random_split_dataset(trainset, num_clients)

    splited_testset = random_split_dataset(testset, num_clients)

    return splited_trainset, splited_testset, norm_layer


def cifar100(
    data_dir, batch_size, mode="base", normalize=True, norm_layer=None, size=32, num_clients=10
):
    """
    mode: org | base | ssl
    """
    transform_train = [
        transforms.Resize(size),
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    transform_test = [transforms.Resize(size), transforms.ToTensor()]

    if mode == "org":
        None
    elif mode == "base":
        transform_train = [transforms.Resize(size), transforms.ToTensor()]
    elif mode == "ssl":
        transform_train = [
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]
    else:
        raise ValueError(f"{mode} mode not supported")

    if norm_layer is None:
        norm_layer = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
        )

    if normalize:
        transform_train.append(norm_layer)
        transform_test.append(norm_layer)

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    if mode == "ssl":
        transform_train = TwoCropTransform(transform_train)

    trainset = datasets.CIFAR100(
        root=os.path.join(data_dir, "cifar100"),
        train=True,
        download=True,
        transform=transform_train,
    )
    testset = datasets.CIFAR100(
        root=os.path.join(data_dir, "cifar100"),
        train=False,
        download=True,
        transform=transform_test,
    )

    splited_trainset = random_split_dataset(trainset, num_clients)

    splited_testset = random_split_dataset(testset, num_clients)

    return splited_trainset, splited_testset, norm_layer


def get_subset(dataset, index):
    result = deepcopy(dataset)
    result.data = dataset.data[index]
    result.targets = list(np.array(dataset.targets)[index])
    if hasattr(result, "neg_img"):
        result.neg_img = dataset.neg_img[index]
    return result


def concat_set(dataset_list):
    result = deepcopy(dataset_list[0])
    result.data = np.concatenate([dataset.data for dataset in dataset_list])
    result.targets = np.concatenate([dataset.targets for dataset in dataset_list])
    if hasattr(result, "neg_img"):
        result.neg_img = np.concatenate([dataset.neg_img for dataset in dataset_list])
    return result


def random_split_dataset(dataset, num):
    dataset_list = []
    splited_size = len(dataset) // num
    for i in range(num - 1):
        new_dataset = deepcopy(dataset)
        new_dataset.data = dataset.data[i * splited_size: (i + 1) * splited_size]
        new_dataset.targets = list(np.array(dataset.targets)[i * splited_size: (i + 1) * splited_size])
        if hasattr(dataset, "neg_img"):
            new_dataset.neg_img = dataset.neg_img[i * splited_size: (i + 1) * splited_size]
        dataset_list.append(new_dataset)
    new_dataset = deepcopy(dataset)
    new_dataset.data = dataset.data[(num - 1) * splited_size:]
    new_dataset.targets = list(np.array(dataset.targets)[(num - 1) * splited_size:])
    if hasattr(dataset, "neg_img"):
        new_dataset.neg_img = dataset.neg_img[(num - 1) * splited_size:]
    dataset_list.append(new_dataset)

    return dataset_list

def get_trainset(trainset, ood_trainset, ratio_pollution):

    normal_index = list(range(len(trainset)))
    np.random.shuffle(normal_index)
    normal_index = normal_index[:int(10000 * (1 - ratio_pollution))]
    anomaly_index = list(range(len(ood_trainset)))
    np.random.shuffle(anomaly_index)
    anomaly_index = anomaly_index[:int(10000 * ratio_pollution)]
    cifar10_set = get_subset(trainset, normal_index)
    cifar100_set = get_subset(ood_trainset, anomaly_index)
    return concat_set([cifar10_set, cifar100_set])

def get_shifted_trainset(trainset, ood_trainset, ratio_pollution):
    shifted_trainset = []

    for i in range(10):
        normal_index = np.argwhere(np.array(trainset.targets) == i).squeeze().tolist()
        np.random.shuffle(normal_index)
        normal_index = normal_index[:int(1000 * (1 - ratio_pollution) * 0.9)]
        extra_normal_index = np.argwhere(np.array(trainset.targets) != i).squeeze().tolist()
        np.random.shuffle(extra_normal_index)
        extra_normal_index = extra_normal_index[:int(1000 * (1 - ratio_pollution) * 0.1)]
        normal_index = normal_index + extra_normal_index
        anomaly_index = list(range(len(ood_trainset)))
        np.random.shuffle(anomaly_index)
        anomaly_index = anomaly_index[:int(1000 * ratio_pollution)]
        cifar10_set = get_subset(trainset, normal_index)
        cifar100_set = get_subset(ood_trainset, anomaly_index)
        shifted_trainset.append(concat_set([cifar10_set, cifar100_set]))

    return shifted_trainset


class MyCIFAR10(datasets.CIFAR10):

    def __init__(self, *args, **kwargs):
        super(MyCIFAR10, self).__init__(*args, **kwargs)

        self.neg_img = np.zeros_like(self.data)
        self.pseudo_labels = np.ones_like(self.targets)

    def __getitem__(self, index):

        img, target, neg_img, pseudo_label = \
            self.data[index], self.targets[index], self.neg_img[index], self.pseudo_labels[index]

        img = Image.fromarray(img)
        neg_img = Image.fromarray(neg_img)

        if self.transform is not None:
            img = self.transform(img)
            neg_img = self.transform(neg_img)

        return img, target, neg_img, pseudo_label, index


class MyCIFAR100(datasets.CIFAR100):

    def __init__(self, *args, **kwargs):
        super(MyCIFAR100, self).__init__(*args, **kwargs)

        self.neg_img = np.zeros_like(self.data)
        self.pseudo_labels = np.ones_like(self.targets)

    def __getitem__(self, index):

        img, target, neg_img, pseudo_label = \
            self.data[index], self.targets[index], self.neg_img[index], self.pseudo_labels[index]

        img = Image.fromarray(img)
        neg_img = Image.fromarray(neg_img)

        if self.transform is not None:
            img = self.transform(img)
            neg_img = self.transform(neg_img)

        return img, target, neg_img, pseudo_label, index


def save_data(trainset_list, testset_list, oodset_list, num_clients, path=None):
    if path is None:
        path = "./data/splited_data/"
    for i in range(num_clients):
        np.save(path + "trainset_img_{}.npy".format(i), trainset_list[i].data)
        np.save(path + "testset_img_{}.npy".format(i), testset_list[i].data)
        np.save(path + "oodset_img_{}.npy".format(i), oodset_list[i].data)
        np.save(path + "trainset_target_{}.npy".format(i), trainset_list[i].targets)
        np.save(path + "testset_target_{}.npy".format(i), testset_list[i].targets)
        np.save(path + "oodset_target_{}.npy".format(i), oodset_list[i].targets)


def load_data(trainset_list, testset_list, oodset_list, num_clients, path=None):
    if path is None:
        path = "./data/splited_data/"
    for i in range(num_clients):
        trainset_list[i].data = np.load(path + "trainset_img_{}.npy".format(i))
        testset_list[i].data = np.load(path + "testset_img_{}.npy".format(i))
        oodset_list[i].data = np.load(path + "oodset_img_{}.npy".format(i))
        trainset_list[i].targets = np.load(path + "trainset_target_{}.npy".format(i))
        testset_list[i].targets = np.load(path + "testset_target_{}.npy".format(i))
        oodset_list[i].targets = np.load(path + "oodset_target_{}.npy".format(i))

    return trainset_list, testset_list, oodset_list