import data
import numpy as np
from utils import *
from eval_ssd import *
from copy import deepcopy
from torch.utils.data import DataLoader

class Evaluator:

    def __init__(self, args):
        self.args = args

    def eval_ssd(self, model, trainset, testset, oodset, args):
        trainset = deepcopy(trainset)
        testset = deepcopy(testset)
        oodset = deepcopy(oodset)
        trainset.transform = testset.transform

        train_loader = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True
        )
        test_loader = DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        ood_loader = DataLoader(
            oodset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

        features_train, labels_train = get_features(model.encoder, train_loader)
        features_test, _ = get_features(model.encoder, test_loader)
        print("In-distribution features shape: ", features_train.shape, features_test.shape)
        features_ood, _ = get_features(model.encoder, ood_loader)
        print("Out-of-distribution features shape: ", features_ood.shape)

        accuracy, precision, recall, f_score, auroc, aupr = get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train),
            self.args,
        )

        return accuracy, precision, recall, f_score, auroc, aupr

    def get_clusters(self, model, trainset, testset, args):

        trainset = deepcopy(trainset)
        testset = deepcopy(testset)
        trainset.transform = testset.transform

        train_loader = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        features_train, _ = get_features(model.encoder, train_loader)

        features_train /= np.linalg.norm(features_train, axis=-1, keepdims=True) + 1e-10

        kmeans = faiss.Kmeans(
            features_train.shape[1], args.clusters, niter=20, verbose=False, gpu=True
        )
        kmeans.train(np.random.permutation(features_train))
        _, ypred = kmeans.assign(features_train)
        xc = [features_train[ypred == i] for i in np.unique(ypred)]

        mean_list = []
        cov_list = []
        for x in xc:
            mean_list.append(np.mean(x, axis=0, keepdims=True))
            cov_list.append(ShrunkCovariance().fit(x).covariance_)

        return mean_list, cov_list

    def set_neg_img(self, model, trainset, testset, mean_list, cov_list, args):

        trainset = deepcopy(trainset)
        testset = deepcopy(testset)

        transform_train = trainset.transform
        trainset.transform = testset.transform

        train_loader = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        neg_imgs, features = \
            get_neg_features(model.encoder, trainset, train_loader, shift_trans_type=args.shift_trans_type)

        num_imgs, num_transform = features.shape[0], features.shape[1]

        features = features.reshape((num_imgs * num_transform, -1))
        features /= np.linalg.norm(features, axis=-1, keepdims=True) + 1e-10

        num_clusters = len(mean_list)

        distances = [
            np.sum(
                (features - mean_list[i])
                * (
                    np.linalg.pinv(cov_list[i]).dot(
                        (features - mean_list[i]).T
                    )
                ).T,
                axis=-1,
            )
            for i in range(num_clusters)
        ]

        distances = np.min(distances, axis=0)
        distances = distances.reshape((num_imgs, num_transform))
        neg_index = np.argmax(distances, 1)

        neg_imgs = neg_imgs[np.arange(0, num_imgs), neg_index]

        trainset.neg_img = neg_imgs

        trainset.transform = transform_train
        return trainset

    def set_neg_labels(self, model, trainset, testset, mean_list, cov_list, args):

        trainset = deepcopy(trainset)
        testset = deepcopy(testset)

        transform_train = trainset.transform
        trainset.transform = testset.transform

        train_loader = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

        features, _ = get_aligned_features(model.encoder, train_loader)
        num_imgs = features.shape[0]

        features /= np.linalg.norm(features, axis=-1, keepdims=True) + 1e-10

        num_clusters = len(mean_list)

        distances = [
            np.sum(
                (features - mean_list[i])
                * (
                    np.linalg.pinv(cov_list[i]).dot(
                        (features - mean_list[i]).T
                    )
                ).T,
                axis=-1,
            )
            for i in range(num_clusters)
        ]

        distances = np.min(distances, axis=0)

        k = int(0.9 * num_imgs)
        pseudo_index = np.argpartition(distances, k)[:k]
        pseudo_labels = np.zeros(num_imgs)
        pseudo_labels[pseudo_index] = 1.0

        trainset.pseudo_labels = pseudo_labels

        trainset.transform = transform_train
        return trainset
