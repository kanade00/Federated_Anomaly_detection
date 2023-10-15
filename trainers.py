import torch
import torch.nn as nn
import time
from utils import AverageMeter, ProgressMeter, accuracy
from shift_transform import get_shift_module


def ssl_csi(
    model,
    device,
    dataloader,
    criterion,
    optimizer,
    lr_scheduler=None,
    epoch=0,
    args=None,
    neg=False
):

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()

    shift_trans, K_shift = get_shift_module(args)
    # K_shift = 2

    for i, data in enumerate(dataloader):
        images, target = data[0], data[1].to(device)
        images1, images2 = images[0].to(device), images[1].to(device)

        if neg:
            # neg_imgs = data[2]
            # neg_imgs1, neg_imgs2 = neg_imgs[0].to(device), neg_imgs[1].to(device)
            # images1 = torch.cat([images1, neg_imgs1])
            # images2 = torch.cat([images2, neg_imgs2])
            pseudo_label = data[3].unsqueeze(0)
        images1 = torch.cat([shift_trans(images1, k) for k in range(K_shift)])
        images2 = torch.cat([shift_trans(images2, k) for k in range(K_shift)])
        images = torch.cat([images1, images2], dim=0)
        bsz = target.shape[0]

        # basic properties of training
        if i == 0:
            print(
                images.shape,
                target.shape,
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )
            print(
                "Pixel range for training images : [{}, {}]".format(
                    torch.min(images).data.cpu().numpy(),
                    torch.max(images).data.cpu().numpy(),
                )
            )

        features = model(images)
        if neg:
            features_list = torch.split(features, [bsz * K_shift] * 2, dim=0)
        else:
            features_list = torch.split(features, [bsz * K_shift] * 2, dim=0)
        features = torch.cat([feature.unsqueeze(1) for feature in features_list], dim=1)
        if args.training_mode == "SupCon":
            loss = criterion(features, target)
        elif args.training_mode == "SimCLR":
            if neg:
                logits_mask = torch.zeros(bsz * K_shift, bsz * K_shift)
                for i in range(K_shift):
                    logits_mask[bsz * i: bsz * (i + 1), bsz * i: bsz * (i + 1)] = pseudo_label * pseudo_label.t()
                logits_mask = 1 - logits_mask
                logits_mask = torch.max(logits_mask, torch.eye(K_shift * bsz, dtype=logits_mask.dtype))
                logits_mask = logits_mask.repeat(2, 2)
                logits_mask = torch.scatter(
                    logits_mask,
                    1,
                    torch.arange(2 * bsz * K_shift).view(-1, 1),
                    0,
                )
                loss = criterion(features, logits_mask=logits_mask)
            else:
                loss = criterion(features)
        else:
            raise ValueError("training mode not supported")

        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses