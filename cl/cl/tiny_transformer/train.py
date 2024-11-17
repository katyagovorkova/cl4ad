import numpy as np
from argparse import ArgumentParser
import os
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
torch.autograd.set_detect_anomaly(True)

class BackgroundDataset(Dataset):
    'Characterizes the background dataset for PyTorch'
    def __init__(self, x, ix, labels, device, augmentation=False, ixa=None):
          'Initialization'
          self.device = device
          self.data = torch.from_numpy(x[ix]).to(dtype=torch.float32, device=self.device)
          self.labels = torch.from_numpy(labels[ix]).to(dtype=torch.long, device=self.device)
          # if augmentation, prepare augmented outputs for vicreg loss
        #   ixa = np.concatenate((ix[1:], ix[:1]))
          self.augmented_data = torch.from_numpy(x[ixa]).to(dtype=torch.float32, device=self.device) if augmentation else None

    def __len__(self):
          'Denotes the total number of samples'
          return len(self.data)

    def __getitem__(self, index):
          'Generates one sample of data'
          if self.augmented_data is not None:
              return self.data[index], self.augmented_data[index], self.labels[index]
          return self.data[index], self.labels[index]


# Simplified implemented from https://github.com/violatingcp/codec/blob/main/losses.py
class SimCLRLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.logical_not(mask).float()

        logits = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        exp_logits += torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - self.temperature * mean_log_prob_pos

        loss = loss.view(1, batch_size).float().mean()

        return loss


class VICRegLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        repr_loss = F.mse_loss(x, y)
        
        x_mu = x.mean(dim=0)
        x_std = torch.sqrt(x.var(dim=0)+1e-4)
        y_mu = y.mean(dim=0)
        y_std = torch.sqrt(y.var(dim=0)+1e-4)

        x = (x - x_mu)
        y = (y - y_mu)

        N = x.size(0)
        D = x.size(-1)

        std_loss = torch.mean(F.relu(1 - x_std)) / 2
        std_loss += torch.mean(F.relu(1 - y_std)) / 2

        cov_x = (x.T.contiguous() @ x) / (N - 1)
        cov_y = (y.T.contiguous() @ y) / (N - 1)

        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(D)
        cov_loss += self.off_diagonal(cov_y).pow_(2).sum().div(D)

        weighted_inv = repr_loss * 25. # * self.hparams.invariance_loss_weight
        weighted_var = std_loss * 25. # self.hparams.variance_loss_weight
        weighted_cov = cov_loss * 1. #self.hparams.covariance_loss_weight

        loss = weighted_inv + weighted_var + weighted_cov

        return loss

    def off_diagonal(self, x):
        #num_batch, n, m = x.shape
        n, m = x.shape
        assert n == m
        # All off diagonal elements from complete batch flattened
        #return x.flatten(start_dim=1)[...,:-1].view(num_batch, n - 1, n + 1)[...,1:].flatten()
        return x.flatten()[...,:-1].view(n - 1, n + 1)[...,1:].flatten()


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')
    print(args.notes)

    torch.manual_seed(0)
    np.random.seed(0)

    dataset = np.load(args.background_dataset)
    model_dir = args.model_dir

    #parameters
    latend_dim = args.latent_dim
    heads = args.heads
    layers = args.layers
    expansion = args.expansion
    alpha = args.alpha
    lr = args.lr
    wd = args.wd
    epochs = args.epochs
    batch_size = args.batch_size

    kfold = True if args.kfold == "true" else False
    vicreg = True if args.vicreg == "true" else False
    simclr = True if args.simclr == "true" else False
    loss_name = "vicreg" if args.vicreg == "true" else "simclr"




if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()

    # If not using full data, name of smaller dataset to pull from
    parser.add_argument('--background-dataset', type=str)
    parser.add_argument('--anomaly-dataset', type=str)

    parser.add_argument('--latent-dim', type=int)
    parser.add_argument('--heads', type=int)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--expansion', type=int, default=16)

    parser.add_argument('--proportioned', type=str)
    parser.add_argument('--kfold', type=str, default='false')
    parser.add_argument('--vicreg', type=str, default='false')
    parser.add_argument('--simclr', type=str, default='false')
    parser.add_argument('--alpha', type=float)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)

    parser.add_argument('--model-dir', type=str, default='output/kfold_exp/')
    parser.add_argument('--acc-dir', type=str, default='output/kfold_accuracies/')

    parser.add_argument('--notes', type=str)

    args = parser.parse_args()
    main(args)