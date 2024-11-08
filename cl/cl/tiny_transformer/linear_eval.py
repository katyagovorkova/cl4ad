import numpy as np
from argparse import ArgumentParser
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from train import TorchCLDataset
from models import CVAE
from tiny_transformer.model import TransformerModel

id = os.getenv('SLURM_JOB_ID')
if id is None:
    id = 'default'


if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()

    # preprocessed & proportioned background
    parser.add_argument('--background-dataset', type=str, default=None)
    # list of pretrained transformer models
    parser.add_argument('--model-dir', type=str, default=None)
        # e.g. /n/home12/ylanaxu/orca/output/kfold_exp/
    parser.add_argument('--model-names')
        # e.g. vicreg_dim8_head4_fold0_56194352.pth,vicreg_dim8_head4_fold0_56194352.pth
    # where to save loss pdf
    parser.add_argument('--loss-dir', type=str, default='output/simple_classifier/')

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)

    parser.add_argument('--notes', type=str)

    args = parser.parse_args()
    main(args)


def main(args):



class AverageMeter(object):
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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res