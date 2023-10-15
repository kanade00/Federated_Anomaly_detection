from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import time
import logging
import argparse
from collections import OrderedDict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.covariance import ShrunkCovariance
import faiss

import torch
import torch.nn as nn

from models import SupResNet, SSLResNet
from utils import (
    get_features,
    get_roc_sklearn,
    get_pr_sklearn,
    get_fpr,
    get_scores_one_cluster,
)
import data

# local utils for SSD evaluation
def get_scores(ftrain, ftest, food, labelstrain, args):
    if args.clusters == 1:
        return get_scores_one_cluster(ftrain, ftest, food)
    else:
        if args.training_mode == "SupCE":
            print("Using data labels as cluster since model is cross-entropy")
            ypred = labelstrain
        else:
            ypred = get_clusters(ftrain, args.clusters)
        return get_scores_multi_cluster(ftrain, ftest, food, ypred)


def get_clusters(ftrain, nclusters):
    kmeans = faiss.Kmeans(
        ftrain.shape[1], nclusters, niter=100, verbose=False, gpu=False
    )
    kmeans.train(np.random.permutation(ftrain))
    _, ypred = kmeans.assign(ftrain)
    return ypred


def get_scores_multi_cluster(ftrain, ftest, food, ypred):
    xc = [ftrain[ypred == i] for i in np.unique(ypred)]

    din = []
    dood = []

    for x in xc:
        mean = np.mean(x, axis=0, keepdims=True)
        cov = ShrunkCovariance().fit(x).covariance_
        din.append(
            np.sum(
                (ftest - mean)
                * (
                    np.linalg.pinv(cov).dot(
                        (ftest - mean).T
                    )
                ).T,
                axis=-1,
            )
        )

        dood.append(
            np.sum(
                (food - mean)
                * (
                    np.linalg.pinv(cov).dot(
                        (food - mean).T
                    )
                ).T,
                axis=-1,
            )
        )

    din = np.min(din, axis=0)
    dood = np.min(dood, axis=0)

    return din, dood


def get_f1_score(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    scores = np.concatenate((xin, xood))
    max_accuracy = 0
    max_precision = 0
    max_recall = 0
    max_f_score = 0
    for ratio in range(45, 56):
        threshold = np.percentile(scores, ratio)
        pred = (scores > threshold).astype(int)
        accuracy = accuracy_score(labels, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(labels, pred, average='binary', zero_division=0)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            max_precision = precision
            max_recall = recall
            max_f_score = f_score
    return max_accuracy, max_precision, max_recall, max_f_score


def get_eval_results(ftrain, ftest, food, labelstrain, args):
    """
    None.
    """
    # standardize data
    ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
    ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10
    food /= np.linalg.norm(food, axis=-1, keepdims=True) + 1e-10

    m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)

    ftrain = (ftrain - m) / (s + 1e-10)
    ftest = (ftest - m) / (s + 1e-10)
    food = (food - m) / (s + 1e-10)

    dtest, dood = get_scores(ftrain, ftest, food, labelstrain, args)

    accuracy, precision, recall, f_score = get_f1_score(dtest, dood)
    auroc, aupr = get_roc_sklearn(dtest, dood), get_pr_sklearn(dtest, dood)
    return accuracy, precision, recall, f_score, auroc, aupr
