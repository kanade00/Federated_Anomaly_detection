import trainers as trainers
import torch
import numpy as np
from models import SupResNet, SSLResNet
from torch.utils.data import DataLoader
from utils import get_features, get_features_ssl, get_aligned_features, get_aligned_features_ssl
import faiss
from sklearn.covariance import ShrunkCovariance
from copy import deepcopy

class FederatedTrainer:

    def __init__(self, args):
        self.trainer = trainers.ssl_csi

        self.device = "cuda"
        if args.training_mode in ["SimCLR", "SupCon"]:
            self.model = SSLResNet(arch=args.arch, out_dim=128).cpu()
        elif args.training_mode == "SupCE":
            self.model = SupResNet(arch=args.arch, num_classes=args.num_classes).cpu()
        else:
            raise ValueError("training mode not supported")

    def train(self, model_dict, trainset, criterion, lr, epoch, args, neg=False):
        self.model.load_state_dict(model_dict)
        self.model = self.model.to(self.device)

        trainset = deepcopy(trainset)

        train_loader = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True
        )
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        for _ in range(args.local_epoch):
            losses = self.trainer(self.model, self.device, train_loader, criterion, optimizer, None, epoch, args, neg)
        self.model = self.model.cpu()
        return losses

    def eval(self, model_dict, testset, oodset, mean_list, cov_list, args):
        self.model.load_state_dict(model_dict)
        self.model = self.model.to(self.device)

        testset = deepcopy(testset)
        oodset = deepcopy(oodset)

        test_loader = DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        ood_loader = DataLoader(
            oodset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

        # features_test, _ = get_features(self.model.encoder, test_loader)
        # features_ood, _ = get_features(self.model.encoder, ood_loader)
        features_test, _ = get_features(self.model, test_loader)
        features_ood, _ = get_features(self.model, ood_loader)

        features_test /= np.linalg.norm(features_test, axis=-1, keepdims=True) + 1e-10
        features_ood /= np.linalg.norm(features_ood, axis=-1, keepdims=True) + 1e-10

        num_clusters = len(mean_list)

        distances_in = [
            np.sum(
                (features_test - mean_list[i])
                * (
                    np.linalg.pinv(cov_list[i]).dot(
                        (features_test - mean_list[i]).T
                    )
                ).T,
                axis=-1,
            )
            for i in range(num_clusters)
        ]
        distances_ood = [
            np.sum(
                (features_ood - mean_list[i])
                * (
                    np.linalg.pinv(cov_list[i]).dot(
                        (features_ood - mean_list[i]).T
                    )
                ).T,
                axis=-1,
            )
            for i in range(num_clusters)
        ]

        distances_in = np.min(distances_in, axis=0)
        distances_ood = np.min(distances_ood, axis=0)

        # distances_in = np.log(np.sum(np.exp(0.5 * np.array(distances_in)), axis=0))
        # distances_ood = np.log(np.sum(np.exp(0.5 * np.array(distances_ood)), axis=0))

        self.model = self.model.cpu()

        return distances_in, distances_ood

    def get_clusters(self, model_dict, trainset, testset, args):
        self.model.load_state_dict(model_dict)
        self.model = self.model.to(self.device)

        trainset = deepcopy(trainset)
        testset = deepcopy(testset)
        trainset.transform = testset.transform

        train_loader = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        # features_train, _ = get_features(self.model.encoder, train_loader)
        features_train, _ = get_features(self.model, train_loader)

        features_train /= np.linalg.norm(features_train, axis=-1, keepdims=True) + 1e-10

        kmeans = faiss.Kmeans(
            features_train.shape[1], args.clusters, niter=5, verbose=False, gpu=False
        )
        kmeans.train(np.random.permutation(features_train))
        _, ypred = kmeans.assign(features_train)
        xc = [features_train[ypred == i] for i in np.unique(ypred)]

        mean_list = []
        cov_list = []
        for x in xc:
            mean_list.append(np.mean(x, axis=0, keepdims=True))
            cov_list.append(ShrunkCovariance().fit(x).covariance_)

        self.model = self.model.cpu()

        return mean_list, cov_list

    def get_clusters_for_train(self, model_dict, trainset, args):
        self.model.load_state_dict(model_dict)
        self.model = self.model.to(self.device)

        trainset = deepcopy(trainset)

        train_loader = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        # features_train, _ = get_features(self.model, train_loader)
        features_train, _ = get_features_ssl(self.model, train_loader)

        features_train /= np.linalg.norm(features_train, axis=-1, keepdims=True) + 1e-10

        kmeans = faiss.Kmeans(
            features_train.shape[1], args.clusters * 2, niter=5, verbose=False, gpu=False
        )
        kmeans.train(np.random.permutation(features_train))
        _, ypred = kmeans.assign(features_train)
        xc = [features_train[ypred == i] for i in np.unique(ypred)]

        mean_list = []
        cov_list = []
        for x in xc:
            mean_list.append(np.mean(x, axis=0, keepdims=True))
            cov_list.append(ShrunkCovariance().fit(x).covariance_)

        self.model = self.model.cpu()

        return mean_list, cov_list

    def set_neg_labels(self, model_dict, trainset, mean_list, cov_list, args):
        self.model.load_state_dict(model_dict)
        self.model = self.model.to(self.device)

        transform_train = trainset.transform

        train_loader = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

        # features, _ = get_aligned_features(self.model.encoder, train_loader)
        features1, features2, _ = get_aligned_features_ssl(self.model, train_loader)
        num_imgs = features1.shape[0]

        features1 /= np.linalg.norm(features1, axis=-1, keepdims=True) + 1e-10
        features2 /= np.linalg.norm(features2, axis=-1, keepdims=True) + 1e-10

        num_clusters = len(mean_list)

        distances1 = [
            np.sum(
                (features1 - mean_list[i])
                * (
                    np.linalg.pinv(cov_list[i]).dot(
                        (features1 - mean_list[i]).T
                    )
                ).T,
                axis=-1,
            )
            for i in range(num_clusters)
        ]

        distances2 = [
            np.sum(
                (features2 - mean_list[i])
                * (
                    np.linalg.pinv(cov_list[i]).dot(
                        (features2 - mean_list[i]).T
                    )
                ).T,
                axis=-1,
            )
            for i in range(num_clusters)
        ]

        # distances1 = np.log(np.sum(np.exp(0.5 * np.array(distances1)), axis=0))
        # distances2 = np.log(np.sum(np.exp(0.5 * np.array(distances2)), axis=0))
        distances1 = np.min(distances1, axis=0)
        distances2 = np.min(distances2, axis=0)

        distances = (distances1 + distances2) / 2

        k = int(0.70 * num_imgs)
        pseudo_index = np.argpartition(distances, k)[:k]
        pseudo_labels = np.zeros(num_imgs)
        pseudo_labels[pseudo_index] = 1.0
        # pseudo_labels = np.ones(num_imgs)

        trainset.pseudo_labels = pseudo_labels

        self.model = self.model.cpu()

        trainset.transform = transform_train
        return trainset

    def get_model_dict(self):
        return self.model.state_dict()