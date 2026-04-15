#!/usr/bin/env python
"""Linear probing evaluation for MoCo pretrained encoders on CT data.

Freezes all layers of a pretrained ResNet-50 backbone and trains only a
linear classification head (fc layer) on a labeled downstream dataset.
This measures the quality of learned representations without fine-tuning
the backbone weights.

Uses the same CT data pipeline as pretraining: loads cached .pt volumes,
extracts 2.5D crops, and applies HU-preserving augmentations.  Labels
are provided via CSV files with ``filename`` and ``label`` columns.

Usage:
    python main_lincls.py \
        --data /path/to/cached-tensors \
        --train-csv labels_train.csv --val-csv labels_val.csv \
        --pretrained /path/to/checkpoint_0199.pth.tar \
        --num-classes 3 --epochs 100

Based on the MoCo reference implementation by Meta AI Research:
    https://github.com/facebookresearch/moco
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import builtins
import os
import random
import shutil
import time
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

from moco.ct_dataset import CTLinClsDataset

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="MoCo Linear Probing on CT Data")
parser.add_argument("--data", required=True, metavar="DIR",
                    help="path to cached .pt tensor directory")
parser.add_argument("--train-csv", required=True,
                    help="CSV with filename,label columns for training split")
parser.add_argument("--val-csv", required=True,
                    help="CSV with filename,label columns for validation split")
parser.add_argument("--num-classes", required=True, type=int,
                    help="number of classification categories")
parser.add_argument(
    "-a", "--arch", metavar="ARCH", default="resnet50", choices=model_names,
    help="model architecture (default: resnet50)",
)
parser.add_argument(
    "-j", "--workers", default=32, type=int, metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs", default=100, type=int, metavar="N",
    help="number of total epochs to run",
)
parser.add_argument(
    "--start-epoch", default=0, type=int, metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b", "--batch-size", default=256, type=int, metavar="N",
    help="mini-batch size (default: 256), total across all GPUs",
)
parser.add_argument(
    "--lr", "--learning-rate", default=30.0, type=float, metavar="LR",
    help="initial learning rate", dest="lr",
)
parser.add_argument(
    "--schedule", default=[60, 80], nargs="*", type=int,
    help="learning rate schedule (when to drop lr by 10x)",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M",
                    help="momentum")
parser.add_argument(
    "--wd", "--weight-decay", default=0.0, type=float, metavar="W",
    help="weight decay (default: 0.)", dest="weight_decay",
)
parser.add_argument(
    "-p", "--print-freq", default=10, type=int, metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument("--resume", default="", type=str, metavar="PATH",
                    help="path to latest lincls checkpoint (default: none)")
parser.add_argument("-e", "--evaluate", dest="evaluate", action="store_true",
                    help="evaluate model on validation set")
parser.add_argument("--world-size", default=-1, type=int,
                    help="number of nodes for distributed training")
parser.add_argument("--rank", default=-1, type=int,
                    help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://224.66.41.62:23456", type=str,
                    help="url used to set up distributed training")
parser.add_argument("--dist-backend", default="nccl", type=str,
                    help="distributed backend")
parser.add_argument("--seed", default=None, type=int,
                    help="seed for initializing training")
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed", action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs.",
)
parser.add_argument("--pretrained", default="", type=str,
                    help="path to moco pretrained checkpoint")
parser.add_argument("--crops-per-volume", default=5, type=int,
                    help="crops per volume per epoch (default: 5)")
parser.add_argument(
    "--output-dir", default=".", type=str,
    help="directory to save checkpoints (default: current directory)",
)

best_acc1 = 0


def main():
    """Parse arguments and launch training workers (one per GPU)."""
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    """Initialize DDP, load pretrained weights, and run linear evaluation."""
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    # Create model with the correct number of output classes
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=args.num_classes)

    # Freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ["fc.weight", "fc.bias"]:
            param.requires_grad = False
    # Init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # Load MoCo pretrained backbone weights
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # Rename moco pre-trained keys: strip "module.encoder_q." prefix,
            # skip the projection head (fc) since we replace it
            state_dict = checkpoint["state_dict"]
            for k in list(state_dict.keys()):
                if k.startswith("module.encoder_q") and not k.startswith(
                    "module.encoder_q.fc"
                ):
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # Only optimize the linear classifier (fc layer)
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias
    optimizer = torch.optim.SGD(
        parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    # Optionally resume from a linear probing checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            if args.gpu is not None:
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # CT data loading — no ImageNet normalization, HU values already in [0, 1]
    train_dataset = CTLinClsDataset(
        args.data, args.train_csv,
        crops_per_volume=args.crops_per_volume, is_train=True,
    )
    val_dataset = CTLinClsDataset(
        args.data, args.val_csv,
        crops_per_volume=args.crops_per_volume, is_train=False,
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        train(train_loader, model, criterion, optimizer, epoch, args)

        acc1 = validate(val_loader, model, criterion, args)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                filename=os.path.join(args.output_dir, "lincls_checkpoint.pth.tar"),
            )
            if epoch == args.start_epoch:
                sanity_check(model.state_dict(), args.pretrained)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch),
    )

    # Eval mode: under the protocol of linear classification on frozen
    # features, BatchNorm must not update running mean/std — those are
    # part of the pretrained model parameters.
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    """Evaluate the model on the validation set and return top-1 accuracy."""
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1], prefix="Test: "
    )

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(" * Acc@1 {top1.avg:.3f}".format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(
            os.path.dirname(filename), "model_best.pth.tar"
        ))


def sanity_check(state_dict, pretrained_weights):
    """Verify that only the fc layer changed — backbone must stay frozen."""
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint["state_dict"]

    for k in list(state_dict.keys()):
        if "fc.weight" in k or "fc.bias" in k:
            continue

        # Map current key to the pretrained checkpoint's naming convention
        k_pre = (
            "module.encoder_q." + k[len("module."):]
            if k.startswith("module.")
            else "module.encoder_q." + k
        )

        assert (
            state_dict[k].cpu() == state_dict_pre[k_pre]
        ).all(), "{} is changed in linear classifier training.".format(k)

    print("=> sanity check passed.")


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
