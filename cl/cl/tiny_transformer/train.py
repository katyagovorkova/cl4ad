import numpy as np
from argparse import ArgumentParser
import os
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from model import TransformerModel
torch.autograd.set_detect_anomaly(True)

id = os.getenv('SLURM_JOB_ID')
if id is None:
    id = 'default'

class BackgroundDataset(Dataset):
    'Characterizes the background dataset for PyTorch'
    def __init__(self, x, ix, weight, labels, ixa=None, augmentation=False):
          'Initialization'
        #   self.device = device
          self.data = torch.from_numpy(x[ix]).to(dtype=torch.float32)
          self.labels = torch.from_numpy(labels[ix]).to(dtype=torch.long)
          self.weights = torch.from_numpy(weight[ix]).to(dtype=torch.float32)
          # if augmentation, prepare augmented outputs for vicreg loss
          if ixa is not None:
            self.augmented_data = torch.from_numpy(x[ixa]).to(dtype=torch.float32) if augmentation else None
            # self.xa_weights = torch.from_numpy(weight[ixa]).to(dtype=torch.float32)
            assert (labels[ix] == labels[ixa]).all()

    def __len__(self):
          'Denotes the total number of samples'
          return len(self.data)

    def __getitem__(self, index):
          'Generates one sample of data'
          if self.augmented_data is not None:
              return self.data[index], self.augmented_data[index], self.labels[index], self.weights[index]
          return self.data[index], self.labels[index], self.x_weights(index)


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
    wandb.login()

    torch.manual_seed(0)
    np.random.seed(0)

    dataset = np.load(args.background_dataset)
    model_dir = args.model_dir

    #parameters
    latent_dim = args.latent_dim
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
    weighted = args.weighted

    cl_loss = VICRegLoss() if vicreg else SimCLRLoss()
    cross_entropy = nn.CrossEntropyLoss(reduction='none')

    wandb.init(
        project="kfold transformer",
        name=f"dim{latent_dim}_head{heads}_{id}",
        config={
            "id": id,
            "weighted": weighted,
            "epochs": args.epochs,
            "lr": lr,
            "wd": wd,
            "batch size": args.batch_size,
            "dim": latent_dim,
            "heads": heads,
            "layers": layers,
            "expansion": expansion,
            "loss": loss_name,
            "notes": args.notes,
        }
    )
    
    def train_epoch(model, data_loader, opt):
        print('training epoch')
        loss_sum = 0.
        count = 0
        for x, x_aug, label, weight in data_loader: 
            opt.zero_grad()
            x = x.squeeze(-1).to(device)
            x_aug = x_aug.squeeze(-1).to(device)
            label = label.to(device)
            weight = weight.to(device)
            outputs = model(x)
            aug_outputs = model(x_aug)
            if vicreg:
                ce = cross_entropy(outputs, label)
                cl = cl_loss(aug_outputs.reshape((-1, latent_dim)), outputs.reshape((-1, latent_dim)))
                loss = ((ce * (1-alpha) + cl * alpha) * weight).sum()
            if simclr:
                loss = cross_entropy(outputs, label) * (1-alpha) + cl_loss(x, label) * alpha
            loss.backward()
            opt.step()

            loss_sum += loss.item()
            count += 1
        return loss_sum/count 
    
    def val_epoch(model, data_loader):
        model.eval()
        true_labels = []
        predictions = []
        loss_sum = 0.
        count = 0

        with torch.no_grad():
            for x, x_aug, label, weight in data_loader:
                x = x.squeeze(-1).to(device)
                x_aug = x_aug.squeeze(-1).to(device)
                label.to(device)
                outputs = model(x)
                aug_outputs = model(x_aug)
                if vicreg:
                    ce = cross_entropy(outputs, label)
                    cl = cl_loss(aug_outputs.reshape((-1, latent_dim)), outputs.reshape((-1, latent_dim)))
                    loss = ((ce * (1-alpha) + cl * alpha) * weight).sum()
                if simclr:
                    loss = cross_entropy(outputs, label) * (1-alpha) + cl_loss(x, label) * alpha
                _, predicted_labels = torch.max(outputs, 1)

                true_labels.extend(label.tolist())
                predictions.extend(predicted_labels.tolist())

                loss_sum += loss.item()
                count += 1

        accuracy = accuracy_score(true_labels, predictions)
        return loss_sum/count, accuracy


    if kfold:
        train_loaders = []
        val_loaders = []
        for i in range(5):
            torch.cuda.empty_cache()
            ix = np.concatenate([dataset[f'ix_train_fold_{j}'] for j in range(5) if j != i])
            ixa = np.concatenate([dataset[f'ixa_train_fold_{j}'] for j in range(5) if j != i])
            train = DataLoader(
                BackgroundDataset(
                    dataset['x_combined'],
                    ix,
                    dataset['weight_combined'],
                    dataset['labels_combined'],
                    # device=device,
                    ixa=ixa,
                    augmentation=True
                ),
                batch_size=batch_size,
                shuffle=False,
                drop_last=True
            )
            train_loaders.append(train)
            val = DataLoader(
                BackgroundDataset(
                    dataset['x_combined'],
                    dataset[f'ix_train_fold_{i}'],
                    dataset['weight_combined'],
                    dataset['labels_combined'],
                    # device=device,
                    ixa=dataset[f'ixa_train_fold_{i}'],
                    augmentation=True 
                ),
                batch_size=batch_size,
                shuffle=False,
                drop_last=True
            )
            val_loaders.append(val)

        train_losses = []
        val_losses = []
        val_accuracy = []
        final_acc = []
        for i, (t, v) in enumerate(zip(train_loaders, val_loaders)):
            print('fold', i)
            model = TransformerModel(3, heads, 4, latent_dim, layers, expansion).to(device)
            opt = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=lr, weight_decay=wd)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

            for epoch in range(1, epochs+1): 
                print('epoch', epoch)
                print('starting training')
                model.train()
                avg_train_loss = train_epoch(model, t, opt)
                train_losses.append(avg_train_loss)

                print('starting validating')
                model.eval()
                avg_val_loss, accuracy = val_epoch(model, v)
                val_losses.append(avg_val_loss)
                val_accuracy.append(accuracy)

                # print(f"Train loss after epoch: {avg_train_loss:.4f}")
                print(f"Train loss after epoch: {avg_train_loss}")
                # print(f"F1/Accuracy after epoch: {f1:.4f}/{accuracy:.4f}")
                print(f"Accuracy after epoch: {accuracy}")

                sched.step()

                wandb.log({f'avg train loss fold {i}': avg_train_loss,
                           f'avg val loss fold {i}': avg_val_loss, f'val acc fold {i}': accuracy})

            final_acc.append(accuracy)
            torch.save(model.state_dict(), f'{model_dir}{loss_name}_dim{latent_dim}_head{heads}_fold{i}_{id}.pth')
            
            wandb.log({'final acc': accuracy})
        print(final_acc)

    else:
        model = TransformerModel(3, heads, 4, latent_dim, layers, expansion)
        opt = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=args.lr, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

        train_data_loader = DataLoader(
            BackgroundDataset(
                dataset['x_train'],
                dataset['ix_train'],
                dataset['ixa_train'],
                dataset['iweight_train'],
                dataset['labels_train'].astype(int),
                device),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True)
        
        val_data_loader = DataLoader(
            BackgroundDataset(
                dataset['x_val'],
                dataset['ix_val'],
                dataset['ixa_val'],
                dataset['iweight_val'],
                dataset['labels_val'].astype(int),
                device),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True)




if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()

    # If not using full data, name of smaller dataset to pull from
    parser.add_argument('--background-dataset', type=str)
    # parser.add_argument('--anomaly-dataset', type=str)

    parser.add_argument('--latent-dim', type=int)
    parser.add_argument('--heads', type=int)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--expansion', type=int, default=16)

    parser.add_argument('--proportioned', type=str)
    parser.add_argument('--weighted', type=str, default="false")
    parser.add_argument('--kfold', type=str, default='false')
    parser.add_argument('--vicreg', type=str, default='false')
    parser.add_argument('--simclr', type=str, default='false')
    parser.add_argument('--alpha', type=float)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)

    parser.add_argument('--model-dir', type=str, default='output/kfold_exp/')
    # parser.add_argument('--acc-dir', type=str, default='output/kfold_accuracies/')

    parser.add_argument('--notes', type=str)

    args = parser.parse_args()
    main(args)