import numpy as np
from argparse import ArgumentParser
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
from torch import nn
from torch.utils.data import DataLoader

from train import TorchCLDataset
from models import CVAE
from tiny_transformer.model import TransformerModel

id = os.getenv('SLURM_JOB_ID')
if id is None:
    id = 'default'


class SoftmaxClassifier(nn.Module):
    def __init__(self, dims):
        super(SoftmaxClassifier, self).__init__()
        self.layer1 = nn.Linear(dims, 50)  # First layer
        self.layer2 = nn.Linear(50, 4)       # Second layer outputs the logits for 4 classes

    def forward(self, x):
        x = torch.relu(self.layer1(x))  # Apply ReLU activation function
        x = self.layer2(x)
        return x
    
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


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    print(args.notes)

    torch.manual_seed(0)
    np.random.seed(0)

    # load dataset
    dataset = np.load(args.background_dataset)

    # load saved cvae model if training on cvae embeddings
    cvae_dir = args.cvae_model_dir
    tf_dir = args.tf_model_dir
    if cvae_dir:
        latent_dim = args.latent_dim
        cvae = CVAE(latent_dim=latent_dim).to(device)
        cvae.load_state_dict(torch.load(cvae_dir))
        cvae.eval()

        model = SoftmaxClassifier(latent_dim).to(device)
    elif tf_dir:
        # Parameters
        input_dim = 3  # Each step has 3 features
        num_heads = args.heads  # Number of heads in the multi-head attention mechanism
        num_classes = 4  # You have four classes
        num_layers = args.layers  # Number of transformer blocks
        latent_dim = args.latent_dim
        expansion = args.expansion
        dropout = args.dropout
        tf = TransformerModel(input_dim, num_heads, num_classes, latent_dim, num_layers, expansion, dropout_rate=dropout, embedding_only=True).to(device)
        tf.load_state_dict(torch.load(tf_dir))

        model = SoftmaxClassifier(latent_dim).to(device)
    else:
        model = SoftmaxClassifier(57).to(device)

    # kyle's linear evaluation model
    model = nn.Linear(latent_dim, 4).to(device)
    model.weight.data.normal_(mean=0.0, std=0.01)
    model.bias.data.zero_()

    # load the datasets for pytorch
    train_data_loader = DataLoader(
        TorchCLDataset(
            dataset['x_train'],
            dataset['ix_train'],
            dataset['ixa_train'],
            dataset['labels_train'].astype(int),
            device),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True)
    
    val_data_loader = DataLoader(
        TorchCLDataset(
            dataset['x_val'],
            dataset['ix_val'],
            dataset['ixa_val'],
            dataset['labels_val'].astype(int),
            device),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True)

    criterion = nn.CrossEntropyLoss().to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # kyle's
    param_groups = [dict(params=model.parameters())]
    optimizer = torch.optim.SGD(param_groups, lr=0.001, weight_decay=1e-6)
    best_acc = argparse.Namespace(top1=0, top5=0)

    train_losses = []
    val_losses = []

    def train_epoch(model, data_loader):
        loss_sum = 0.
        count = 0
        for x, _, label in data_loader:  #_ is augmented value
            optimizer.zero_grad()
            # if using cvae embeddings
            if cvae_dir:
                x = cvae(x)
            if tf_dir:
                x = x.squeeze(-1)
                x = tf(x)

            outputs = model(x)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

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
            for x, _, label in data_loader:
                # if using cvae embeddings
                if cvae_dir:
                    x = cvae(x)
                if tf_dir:
                    x = x.squeeze(-1)
                    x = tf(x)

                outputs = model(x)
                loss = criterion(outputs, label)
                _, predicted_labels = torch.max(outputs, 1)

                true_labels.extend(label.tolist())
                predictions.extend(predicted_labels.tolist())

                loss_sum += loss.item()
                count += 1
                
                #kyle
                acc1, acc5, = accuracy(outputs, label, topk=(1,2))
                top1.update(acc1[0].item(), x.size(0))
                top5.update(acc5[0].item(), x.size(0))
        #kyle
        best_acc.top1 = max(best_acc.top1, top1.avg)
        best_acc.top5 = max(best_acc.top5, top5.avg)
        # Calculate metrics
        # accuracy = accuracy_score(true_labels, predictions)
        # precision = precision_score(true_labels, predictions, average='macro')
        # recall = recall_score(true_labels, predictions, average='macro')
        # f1 = f1_score(true_labels, predictions, average='macro')

        # return loss_sum/count, accuracy, precision, recall, f1
        # kyle
        return loss_sum/count, top1.avg, top5.avg,  best_acc.top1, best_acc.top5
    

    train_losses = []
    val_f1 = []
    val_accuracy = []
    #kyle
    best_acc1 = []
    best_acc5 = []
    for epoch in range(1, args.epochs+1):
        print(f'EPOCH {epoch}')
        # Gradient tracking
        avg_train_loss = train_epoch(model, train_data_loader)
        train_losses.append(avg_train_loss)

        top1 = AverageMeter("Acc@1")
        top5 = AverageMeter("Acc@5")
        # avg_val_loss, accuracy, precision, recall, f1 = val_epoch(model, val_data_loader)
        # kyle
        avg_val_loss, avg_top1, avg_top5, best_top1, best_top5 = val_epoch(model, val_data_loader)
        # val_f1.append(f1)
        # val_accuracy.append(accuracy)
        # kyle
        val_losses.append(avg_val_loss)
        best_acc1.append(best_top1)
        best_acc5.append(best_top5)

        print(f"Train loss after epoch: {avg_train_loss:.4f}")
        # print(f"F1/Accuracy after epoch: {f1:.4f}/{accuracy:.4f}")
        print(f"Avg 1 & 5 accuracies after epoch: {avg_top1:.4f}, {avg_top5:.4f}")
        print(f"Best 1 & 5 accuracies after epoch: {best_top1:.4f}, {best_top5:.4f}")
        print()

        scheduler.step()


    # Create a figure with three subplots, arranged horizontally
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))  # 1 row, 3 columns, width 18 inches, height 5 inches

    # Plotting Training Loss
    axes[0].plot(train_losses, label='Train Loss', color='blue')
    axes[0].plot(val_losses, label='Val Loss', color='orange')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].xaxis.set_major_locator(ticker.MultipleLocator(10))
    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[0].legend()

    # Plotting Validation Accuracy
    # axes[1].plot(val_accuracy, label='Validation Accuracy', color='green')
    # kyle
    axes[1].plot(best_acc1, label='Best Validation Accuracy', color='green')
    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(10))
    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[1].legend()

    # Plotting Validation F1 Score
    # axes[2].plot(val_f1, label='Validation F1 Score', color='red')
    # axes[2].set_title('Validation F1 Score')
    # axes[2].set_xlabel('Epochs')
    # axes[2].set_ylabel('F1 Score')
    # axes[2].xaxis.set_major_locator(ticker.MultipleLocator(10))
    # axes[2].yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    # axes[2].grid(True, which='both', linestyle='--', linewidth=0.5)
    # axes[2].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    output_path = f'{args.loss_dir}metrics_{id}.pdf'
    plt.savefig(output_path)



if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()

    # preprocessed & proportioned background
    parser.add_argument('--background-dataset', type=str, default=None)
    # pretrained cvae model
    parser.add_argument('--cvae-model-dir', type=str, default=None)
    # pretrained transformer model
    parser.add_argument('--tf-model-dir', type=str, default=None)
    # where to save loss pdf
    parser.add_argument('--loss-dir', type=str, default='output/simple_classifier/')

    parser.add_argument('--latent-dim', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)

    parser.add_argument('--heads', type=int, default=3)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--expansion', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--notes', type=str)

    args = parser.parse_args()
    main(args)