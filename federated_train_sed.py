# Some part borrowed from official tutorial https://github.com/pytorch/examples/blob/master/imagenet/main.py
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import argparse
import importlib
import time
import logging
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from federated_client import FederatedTrainer
from knowledge_transfer_sed import SEED, KnowledgeTransfer, get_public_dataset

from models_sed import SupResNet, SSLResNet
import splited_data

from losses import SupConLoss
from utils import *
from utils import (
    get_features,
    get_roc_sklearn,
    get_pr_sklearn,
)
from eval_ssd import get_f1_score
import random

import ray
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


def main():
    parser = argparse.ArgumentParser(description="FADngs evaluation")

    parser.add_argument(
        "--results-dir",
        type=str,
        default="./result",
    )  # change this
    parser.add_argument("--exp-name", type=str, default="federated_FADngs_1")
    parser.add_argument(
        "--training-mode", type=str, default="SimCLR", choices=("SimCLR", "SupCon", "SupCE")
    )

    # model
    parser.add_argument("--arch", type=str, default="resnet34", choices=("resnet18", "resnet34", "resnet50", "resnet101"))
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=10, help="number of clusters in k-means clustering")
    parser.add_argument("--ratio_pollution", type=float, default=0.01, help="ratio of anomaly in datasets")

    # training
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--ood_dataset", type=str, default="cifar100")
    parser.add_argument("--data-dir", type=str, default="/home/dongboyu/FADngs/data")
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup", action="store_true", default=True)

    # ssl
    parser.add_argument(
        "--method", type=str, default="SimCLR", choices=["SupCon", "SimCLR", "SupCE"]
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument('--shift_trans_type', help='shifting transformation type', default='rotation',
                        choices=['rotation', 'cutperm', 'none'], type=str)

    # misc
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=50)
    parser.add_argument("--ckpt", type=str, help="checkpoint path")
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument('--num_clients', type=int, default=10,
                        help='number of clients')
    parser.add_argument('--num_each_turn', type=int, default=5,
                        help='number of clients')
    parser.add_argument('--activated_clients_per_turn', type=int, default=10,
                        help='number of activated clients each turn')
    parser.add_argument('--global_lr', type=float, default=1.0,
                        help='global learning rate')
    parser.add_argument('--local_epoch', type=int, default=1,
                        help='Max number of local training epochs in each rounds')

    args = parser.parse_args()
    device = "cuda"

    if args.batch_size > 256 and not args.warmup:
        warnings.warn("Use warmup training for larger batch-sizes > 256")

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    # create resutls dir (for logs, checkpoints, etc.)
    result_main_dir = os.path.join(args.results_dir, args.exp_name)

    if os.path.exists(result_main_dir):
        n = len(next(os.walk(result_main_dir))[-2])  # prev experiments with same name
        result_sub_dir = os.path.join(
            result_main_dir,
            "{}--dataset-{}-arch-{}-lr-{}_epochs-{}".format(
                n + 1, args.dataset, args.arch, args.lr, args.epochs
            ),
        )
    else:
        os.mkdir(result_main_dir)
        result_sub_dir = os.path.join(
            result_main_dir,
            "1--dataset-{}-arch-{}-lr-{}_epochs-{}".format(
                args.dataset, args.arch, args.lr, args.epochs
            ),
        )

    create_subdirs(result_sub_dir)

    # add logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join(result_sub_dir, "setup.log"), "a")
    )
    logger.info(args)

    # seed cuda
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Create model
    if args.training_mode in ["SimCLR", "SupCon"]:
        global_model = SSLResNet(arch=args.arch, out_dim=128).cpu()
    elif args.training_mode == "SupCE":
        global_model = SupResNet(arch=args.arch, num_classes=args.num_classes).cpu()
    else:
        raise ValueError("training mode not supported")

    # state = torch.load(os.path.join(result_main_dir, "latest_exp", "checkpoint", "checkpoint.pth.tar"))
    # global_model.load_state_dict(state["state_dict"])

    # Dataloader
    trainset_list, testset_list, norm_layer = splited_data.__dict__[args.dataset](
        args.data_dir,
        ood_dataset=args.ood_dataset,
        mode="ssl" if args.training_mode in ["SimCLR", "SupCon"] else "org",
        normalize=args.normalize,
        size=args.size,
        ratio_pollution=args.ratio_pollution,
        num_clients=args.num_clients
    )

    _, oodset_list, _ = splited_data.ood_dataset(
        args.data_dir,
        args.ood_dataset,
        mode="base",
        normalize=args.normalize,
        norm_layer=norm_layer,
        size=args.size,
    )

    splited_data.save_data(trainset_list, testset_list, oodset_list, args.num_clients)
    # trainset_list, testset_list, oodset_list = \
    #     splited_data.load_data(trainset_list, testset_list, oodset_list, args.num_clients, path="./data/splited_data/")

    public_dataset = get_public_dataset(args)

    criterion = (
        SupConLoss(temperature=args.temperature).cuda()
        if args.training_mode in ["SimCLR", "SupCon"]
        else nn.CrossEntropyLoss().cuda()
    )
    optimizer = torch.optim.SGD(
        global_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    ray.init(num_cpus=8, num_gpus=2, log_to_driver=False)
    Client = ray.remote(num_gpus=0.3)(FederatedTrainer)
    server = Client.remote(args)
    client_list = [Client.remote(args) for _ in range(args.num_each_turn)]


    # warmup
    if args.warmup:
        wamrup_epochs = 100
        print(f"Warmup training for {wamrup_epochs} epochs")
        warmup_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=0.01,
            max_lr=args.lr,
            step_size_up=wamrup_epochs,
        )
        global_model_dict = global_model.state_dict()

        num_clients = args.num_clients
        model_dict_list = [global_model_dict for _ in range(num_clients)]

        for epoch in range(wamrup_epochs):
            lr = warmup_lr_scheduler.get_last_lr()[-1]
            global_model_dict = federated_train(args, global_model_dict, model_dict_list, epoch, criterion,
                                                lr, trainset_list, client_list, False, public_dataset)
            warmup_lr_scheduler.step()
        global_model.load_state_dict(global_model_dict)


    # best_prec1 = 0
    best_auroc = 0

    for p in optimizer.param_groups:
        p["lr"] = args.lr
        p["initial_lr"] = args.lr
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, 1e-5
    )

    global_model_dict = global_model.state_dict()

    num_clients = args.num_clients
    model_dict_list = [global_model_dict for _ in range(num_clients)]

    # for epoch in range(1, 800):
    #     lr_scheduler.step()

    for epoch in range(0, args.epochs + 1):

        lr = lr_scheduler.get_last_lr()[-1]

        print("lr: {}".format(lr))

        if epoch % 10 == 0:
            trainset_list = set_neg_labels(global_model_dict, trainset_list, client_list, args)

        global_model_dict = federated_train(args, global_model_dict, model_dict_list, epoch, criterion,
                                            lr, trainset_list, client_list, True, public_dataset)
        lr_scheduler.step()

        if epoch % 10 == 0:
            accuracy, precision, recall, f_score, auroc, aupr = \
                eval_ssd(global_model_dict, trainset_list, testset_list, oodset_list, client_list, args)

            is_best = auroc > best_auroc
            best_auroc = max(auroc, best_auroc)

            d = {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": global_model_dict,
                "best_auroc": best_auroc,
                "optimizer": optimizer.state_dict(),
            }

            save_checkpoint(
                d,
                is_best,
                os.path.join(result_sub_dir, "checkpoint"),
            )

            if not (epoch + 1) % args.save_freq:
                save_checkpoint(
                    d,
                    is_best,
                    os.path.join(result_sub_dir, "checkpoint"),
                    filename=f"checkpoint_{epoch+1}.pth.tar",
                )

            logger.info(
                f"Epoch {epoch}, validation accuracy = {accuracy}, precision = {precision}, recall = {recall}, f_score = {f_score}\n"
                f"AUROC = {auroc}, AUPR = {aupr}, best_auroc {best_auroc}"
            )

        # clone results to latest subdir (sync after every epoch)
        clone_results_to_latest_subdir(
            result_sub_dir, os.path.join(result_main_dir, "latest_exp")
        )


def federated_train(args, global_model_dict, model_dict_list, epoch, criterion, lr, trainset_list, client_list, neg, public_loader):
    selected_id = random.sample(range(args.num_clients), args.activated_clients_per_turn)
    train_model_list = []
    result_list = []
    for i in range(int(args.activated_clients_per_turn / args.num_each_turn)):
        result_list += ray.get([client_list[j].train.remote(global_model_dict,
                                                            trainset_list[
                                                                selected_id[i * args.num_each_turn + j]],
                                                            criterion, lr, epoch, args, neg
                                                            )
                                for j in range(args.num_each_turn)])
        train_model_list += ray.get([client.get_model_dict.remote() for client in client_list])
    sum_loss = 0
    for i in range(args.activated_clients_per_turn):
        model_dict_list[selected_id[i]] = train_model_list[i]
        sum_loss += result_list[i].avg
    global_model_dict = fed_avg(args, model_dict_list, global_model_dict)
    global_model_dict = esemble_distillation(args, model_dict_list, global_model_dict, public_loader, 3 * lr)

    avg_loss = sum_loss / args.activated_clients_per_turn
    print("Epoch: [{}]\tLoss: {:.3f}".format(epoch, avg_loss))

    return global_model_dict

def esemble_distillation(args, model_dict_list, global_model_dict, public_dataset, lr):

    seed_model = SEED(model_dict_list, global_model_dict, args, dim=128)
    transfer = KnowledgeTransfer(public_dataset, seed_model)
    # transfer.adapt()
    global_model_dict = transfer.train(lr, args)

    return global_model_dict

def fed_avg(args, model_dict_list, global_model_dict):

    for param_tensor in global_model_dict:
        global_param = global_model_dict[param_tensor]
        sum_param = 0
        for i in range(0, args.activated_clients_per_turn):
            sum_param += model_dict_list[i][param_tensor] - global_param
        param_diff = torch.true_divide(sum_param, args.activated_clients_per_turn)
        global_model_dict[param_tensor] = global_param + args.global_lr * param_diff
    return global_model_dict

def get_mu_cov(global_model_dict, trainset_list, testset_list, client_list, args):
    mu_cov_list = []
    for i in range(int(args.num_clients / args.num_each_turn)):
        mu_cov_list += ray.get([client_list[j].get_clusters.remote(global_model_dict,
                                                                   trainset_list[i * args.num_each_turn + j],
                                                                   testset_list[i * args.num_each_turn + j],
                                                                   args)
                                for j in range(args.num_each_turn)])

    mu_list = []
    cov_list = []
    for i in range(args.num_clients):
        mu_list += mu_cov_list[i][0]
        cov_list += mu_cov_list[i][1]
    return mu_list, cov_list

def get_mu_cov_for_train(global_model_dict, trainset_list, client_list, args):
    mu_cov_list = []
    for i in range(int(args.num_clients / args.num_each_turn)):
        mu_cov_list += ray.get([client_list[j].get_clusters_for_train.remote(global_model_dict,
                                                                             trainset_list[i * args.num_each_turn + j],
                                                                             args)
                                for j in range(args.num_each_turn)])

    mu_list = []
    cov_list = []
    for i in range(args.num_clients):
        mu_list += mu_cov_list[i][0]
        cov_list += mu_cov_list[i][1]
    return mu_list, cov_list

def eval_ssd(global_model_dict, trainset_list, testset_list, oodset_list, client_list, args):
    mu_list, cov_list = get_mu_cov(global_model_dict, trainset_list, testset_list, client_list, args)
    result_list = []
    for i in range(int(args.num_clients / args.num_each_turn)):
        result_list += ray.get([client_list[j].eval.remote(global_model_dict,
                                                           testset_list[i * args.num_each_turn + j],
                                                           oodset_list[i * args.num_each_turn + j],
                                                           mu_list,
                                                           cov_list,
                                                           args)
                                for j in range(args.num_each_turn)])

    distances_in = []
    distances_ood = []
    for i in range(args.num_clients):
        distances_in = np.append(distances_in, result_list[i][0])
        distances_ood = np.append(distances_ood, result_list[i][1])

    accuracy, precision, recall, f_score = get_f1_score(distances_in, distances_ood)
    auroc = get_roc_sklearn(distances_in, distances_ood)
    aupr = get_pr_sklearn(distances_in, distances_ood)
    return accuracy, precision, recall, f_score, auroc, aupr

def set_neg_labels(global_model_dict, trainset_list, client_list, args):
    mu_list, cov_list = get_mu_cov_for_train(global_model_dict, trainset_list, client_list, args)

    result_list = []
    for i in range(int(args.num_clients / args.num_each_turn)):
        result_list += ray.get([client_list[j].set_neg_labels.remote(global_model_dict,
                                                                     trainset_list[i * args.num_each_turn + j],
                                                                     mu_list,
                                                                     cov_list,
                                                                     args)
                                for j in range(args.num_each_turn)])


    return result_list

if __name__ == "__main__":
    main()
