import numpy as np
from argparse import ArgumentParser
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import re
import wandb

import torch
from torch import nn
from torch.utils.data import DataLoader

from train import BackgroundDataset
from model import TransformerModel

id = os.getenv('SLURM_JOB_ID')
if id is None:
    id = 'default'

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


class CombinedModel(nn.Module):
    def __init__(self, transformer, linear_eval):
        super(CombinedModel, self).__init__()
        self.transformer = transformer
        self.linear_eval = linear_eval

    def forward(self, x):
        # Pass through transformer (feature extraction)
        with torch.no_grad():  # Prevent gradient computation for transformer
            features = self.transformer(x)
        # Pass through linear_eval (trainable)
        output = self.linear_eval(features)
        return output


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


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')
    print(args.notes)
    wandb.login()

    torch.manual_seed(0)
    np.random.seed(0)

    dataset = np.load(args.background_dataset)
    model_prefix = args.model_dir
    suffix = args.model_names.split(',')
    model_suffix = []
    for name in suffix:
        prefix, identifier = name.rsplit('_', 1)
        for fold in range(5):  # Folds 0 to 4 inclusive
            # Add fold information before the identifier
            model_suffix.append(f"{prefix}_fold{fold}_{identifier}.pth")
    print(model_suffix)

    for suffix in model_suffix:
        pattern = r"dim(\d+)_head(\d+)_fold(\d+)"
        match = re.search(pattern, suffix)
        if match is None: raise ValueError("dim & head not found")
        vicreg = suffix[:6] == "vicreg"
        simclr = suffix[:6] == "simclr"
        loss_name = "vicreg" if vicreg else "simclr"

        dim = int(match.group(1)) 
        heads = int(match.group(2))  
        fold = int(match.group(3))
        layers = 4
        expansion = 16
        print(f"dim: {dim}, head: {heads}, fold: {fold}")

        with wandb.init(
        project="kyle linear eval",
        group=f"dim{dim}_head{heads}",  # Group by dim and heads
        name=f"dim{dim}_head{heads}_fold{fold}_{id}",
        tags=[f"dim{dim}", f"head{heads}", f"fold{fold}"],
        config={
            "id": id,
            "epochs": args.epochs,
            "batch size": args.batch_size,
            "notes": args.notes,
            "dim": dim,
            "heads": heads,
            "fold": fold,
            "layers": layers,
            "expansion": expansion,
            "loss": loss_name,
        }
        ) as run:
            # # Use dim and heads as the identifier for grouping in wandb
            # wandb.run.name = f"dim{dim}_head{heads}_fold{fold}_{id}"
            # wandb.run.group = f"dim{dim}_head{heads}"  # Group by dim and heads
            # wandb.run.tags = [f"dim{dim}", f"head{heads}", f"fold{fold}"]

            # # Log some initial information
            # wandb.config.update({
            #     "dim": dim,
            #     "heads": heads,
            #     "fold": fold,
            #     "layers": layers,
            #     "expansion": expansion,
            #     "loss": loss_name,
            # })

            tf = TransformerModel(3, heads, 4, dim, layers, expansion, dropout_rate=0.1, embedding_only=True).to(device)
            tf.load_state_dict(torch.load(model_prefix + suffix))

            for param in tf.parameters():
                param.requires_grad = False

            linear_eval = nn.Linear(dim, 4).to(device)
            linear_eval.weight.data.normal_(mean=0.0, std=0.01)
            linear_eval.bias.data.zero_()

            combined_model = CombinedModel(tf, linear_eval)
            combined_model.to(device)

            train_data_loader = DataLoader(
                BackgroundDataset(
                    dataset['x_train'],
                    dataset['ix_train'],
                    dataset['labels_train'],
                    device=device,
                    augmentation=vicreg,
                    ixa=dataset['ixa_train'],),
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=True)
            
            val_data_loader = DataLoader(
                BackgroundDataset(
                    dataset['x_test'],
                    dataset['ix_test'],
                    dataset['labels_test'],
                    device=device,
                    augmentation=vicreg,
                    ixa=dataset['ixa_test'],),
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=True)
            
            criterion = nn.CrossEntropyLoss().to(device=device)
            param_groups = [dict(params=linear_eval.parameters())]
            optimizer = torch.optim.SGD(param_groups, lr=0.001, weight_decay=1e-6)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
            best_acc = argparse.Namespace(top1=0, top5=0)

            train_losses = []
            val_losses = []

            def train_epoch(combined_model, data_loader):
                #TODO: incorporate for simcl
                combined_model.train()
                loss_sum = 0.
                count = 0
                for x, _, label in data_loader:  #_ is augmented value
                    
                    x = x.squeeze(-1)
                    outputs = combined_model(x)
                    loss = criterion(outputs, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    loss_sum += loss.item()
                    count += 1
                return loss_sum/count 

            def val_epoch(combined_model, data_loader):
                combined_model.eval()
                true_labels = []
                predictions = []
                loss_sum = 0.
                count = 0

                with torch.no_grad():
                    for x, _, label in data_loader:

                        x = x.squeeze(-1)
                        outputs = combined_model(x)
                        loss = criterion(outputs, label)
                        _, predicted_labels = torch.max(outputs, 1)

                        true_labels.extend(label.tolist())
                        predictions.extend(predicted_labels.tolist())

                        loss_sum += loss.item()
                        count += 1
                        
                        acc1, acc5, = accuracy(outputs, label, topk=(1,2))
                        top1.update(acc1[0].item(), x.size(0))
                        top5.update(acc5[0].item(), x.size(0))

                best_acc.top1 = max(best_acc.top1, top1.avg)
                best_acc.top5 = max(best_acc.top5, top5.avg)
                return loss_sum/count, top1.avg, top5.avg,  best_acc.top1, best_acc.top5

            train_losses = []
            best_acc1 = []
            best_acc5 = []
            for epoch in range(args.epochs):
                print(f'EPOCH {epoch}')
                avg_train_loss = train_epoch(combined_model, train_data_loader)
                train_losses.append(avg_train_loss)

                top1 = AverageMeter("Acc@1")
                top5 = AverageMeter("Acc@5")
                avg_val_loss, avg_top1, avg_top5, best_top1, best_top5 = val_epoch(combined_model, val_data_loader)
                val_losses.append(avg_val_loss)
                best_acc1.append(best_top1)
                best_acc5.append(best_top5)

                print(f"Train loss after epoch: {avg_train_loss:.4f}")
                print(f"Avg 1 & 5 accuracies after epoch: {avg_top1:.4f}, {avg_top5:.4f}")
                print(f"Best 1 & 5 accuracies after epoch: {best_top1:.4f}, {best_top5:.4f}")
                print()
                wandb.log({"train loss": avg_train_loss, "avg 1 accuracy": avg_top1})

                scheduler.step()


if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()

    # preprocessed & proportioned background
    parser.add_argument('--background-dataset', type=str, default=None)
    # list of pretrained transformer models
    parser.add_argument('--model-dir', type=str, default=None)
        # e.g. /n/home12/ylanaxu/orca/output/kfold_exp/
    parser.add_argument('--model-names', type=str, default=None)
        # e.g. vicreg_dim8_head4_fold0_56194352.pth,vicreg_dim8_head4_fold0_56194352.pth

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)

    parser.add_argument('--notes', type=str)

    args = parser.parse_args()
    main(args)

