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
from federated_client import FederatedTrainer

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from shift_transform import get_shift_module

from models import SupResNet, SSLResNet
import data
import trainers
from losses import SupConLoss
from utils import *
from evaluator import Evaluator
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def main():
    parser = argparse.ArgumentParser(description="FADgde evaluation")

    parser.add_argument(
        "--results-dir",
        type=str,
        default="./result",
    )  # change this
    parser.add_argument("--exp-name", type=str, default="FADgde")
    parser.add_argument(
        "--training-mode", type=str, default="SimCLR", choices=("SimCLR", "SupCon", "SupCE")
    )

    # model
    parser.add_argument("--arch", type=str, default="resnet34")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--ratio_pollution", type=float, default=0.01)

    # training
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--ood_dataset", type=str, default="cifar100")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup", action="store_true", default=True)

    # ssl
    parser.add_argument(
        "--method", type=str, default="SimCLR", choices=["SupCon", "SimCLR", "SupCE"]
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument('--shift_trans_type', help='shifting transformation type', default='cutperm',
                        choices=['rotation', 'cutperm', 'none'], type=str)

    # misc
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=50)
    parser.add_argument("--ckpt", type=str, help="checkpoint path")
    parser.add_argument("--seed", type=int, default=12345)

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
        result_sub_dir = result_sub_dir = os.path.join(
            result_main_dir,
            "{}--dataset-{}-arch-{}-lr-{}_epochs-{}".format(
                n + 1, args.dataset, args.arch, args.lr, args.epochs
            ),
        )
    else:
        os.mkdir(result_main_dir)
        result_sub_dir = result_sub_dir = os.path.join(
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
        model = SSLResNet(arch=args.arch).to(device)
    elif args.training_mode == "SupCE":
        model = SupResNet(arch=args.arch, num_classes=args.num_classes).to(device)
    else:
        raise ValueError("training mode not supported")

    # load feature extractor on gpu
    # model.encoder = torch.nn.DataParallel(model.encoder).to(device)

    # Dataloader
    trainset, testset, norm_layer = data_v2.__dict__[args.dataset](
        args.data_dir,
        mode="ssl" if args.training_mode in ["SimCLR", "SupCon"] else "org",
        normalize=args.normalize,
        size=args.size,
        batch_size=args.batch_size,
        ratio_pollution=args.ratio_pollution
    )

    _, oodset, _ = data_v2.__dict__[args.ood_dataset](
        args.data_dir,
        args.batch_size,
        mode="base",
        normalize=args.normalize,
        norm_layer=norm_layer,
        size=args.size,
    )

    train_loader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True
    )

    criterion = (
        SupConLoss(temperature=args.temperature).cuda()
        if args.training_mode in ["SimCLR", "SupCon"]
        else nn.CrossEntropyLoss().cuda()
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # select training and validation methods
    trainer = trainers_v2.ssl_csi

    val = knn if args.training_mode in ["SimCLR", "SupCon"] else baseeval
    evaluator = Evaluator(args)

    # warmup
    if args.warmup:
        wamrup_epochs = 100
        print(f"Warmup training for {wamrup_epochs} epochs")
        warmup_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=0.01,
            max_lr=args.lr,
            step_size_up=wamrup_epochs * len(train_loader),
        )
        for epoch in range(wamrup_epochs):
            trainer(
                model,
                device,
                train_loader,
                criterion,
                optimizer,
                warmup_lr_scheduler,
                epoch,
                args,
                False
            )

    # best_prec1 = 0
    best_auroc = 0

    for p in optimizer.param_groups:
        p["lr"] = args.lr
        p["initial_lr"] = args.lr
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs * len(train_loader), 1e-5
    )

    for epoch in range(0, args.epochs+1):
        if epoch % 5 == 0:
            mean_list, cov_list = evaluator.get_clusters(model, trainset, testset, args)
            trainset = evaluator.set_neg_labels(model, trainset, testset, mean_list, cov_list, args)

        trainer(
            model, device, train_loader, criterion, optimizer, lr_scheduler, epoch, args, True
        )


        if epoch % 5 == 0:

            accuracy, precision, recall, f_score, auroc, aupr = evaluator.eval_ssd(model, trainset, testset, oodset, args)


            is_best = auroc > best_auroc
            best_auroc = max(auroc, best_auroc)

            d = {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
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

if __name__ == "__main__":
    main()
