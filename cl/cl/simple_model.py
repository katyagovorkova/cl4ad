import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

import torch
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
from models import CVAE


class Classifier(nn.Module):
        def __init__(self, latent_dim, num_classes):
            super(Classifier, self).__init__()
            self.fc = nn.Linear(latent_dim, num_classes)

        def forward(self, x):
            return self.fc(x)
            # self.fc(x)
            # return nn.functional.softmax(x, dim=1)
        
    

class TorchCLDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, features, ix, ixa, labels, criterion, device):
          'Initialization'
          self.device = device
          self.features = torch.from_numpy(features[ix]).to(dtype=torch.float32, device=self.device)
          self.augmentations = torch.from_numpy(features[ixa].copy()).to(dtype=torch.float32, device=self.device)
          self.labels = torch.from_numpy(labels[ix]).to(dtype=torch.float32, device=self.device)

    def __len__(self):
          'Denotes the total number of samples'
          return len(self.features)

    def __getitem__(self, index):
          'Generates one sample of data'
          # Load data and get label
          X = self.features[index]
          X_aug = self.augmentations[index]
          y = self.labels[index]

          return X, X_aug, y


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    if args.mode == "cvae":
        latent_dim = 6  # from models.py
    else:
        latent_dim = 57
    num_classes = 4
    classifier = Classifier(latent_dim, num_classes).to(device)
    summary(classifier, input_size=(latent_dim,))

    dataset = np.load(args.background_dataset)

    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_data_loader = DataLoader(
        TorchCLDataset(
            dataset['x_train'],
            dataset['ix_train'],
            dataset['ixa_train'],
            dataset['labels_train'],
            criterion, device),
        batch_size=args.batch_size,
        shuffle=False)

    val_data_loader = DataLoader(
        TorchCLDataset(
            dataset['x_val'],
            dataset['ix_val'],
            dataset['ixa_val'],
            dataset['labels_val'],
            criterion, device),
        batch_size=args.batch_size,
        shuffle=False)
    


    # Assuming the CVAE class is defined and the path to the model file
    model_path = args.model_name
    cvae = CVAE().to(device)  # Make sure to initialize it with the correct parameters
    cvae.load_state_dict(torch.load(model_path))
    cvae.eval()  # Set the model to evaluation mode if you're only doing inference


    def train_one_epoch(epoch):
        classifier.train()
        running_loss = 0.
        last_loss = 0.

        for idx, (data, _, labels) in enumerate(train_data_loader, 1):
            if data.shape[0] != args.batch_size: continue
            labels = labels.long()

            data, labels = data.to(device), labels.to(device)  # Move data to the correct device
            if args.mode == 'cvae':
                z_proj = cvae(data)  # This will get z_proj directly using cvae.forward
            else: 
                    z_proj = data

            # Pass z_proj to classifier
            class_preds = classifier(z_proj)
            loss = criterion(class_preds, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if idx % 500 == 0:
                last_loss = running_loss / 500
                running_loss = 0

        return last_loss

    def val_one_epoch(epoch):
            classifier.eval()
            val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for idx, (data, _, labels) in enumerate(val_data_loader, 1):
                    labels = labels.long()
                    if data.shape[0] != args.batch_size: continue

                    data, labels = data.to(device), labels.to(device)  # Move data to the correct device
                    if args.mode == 'cvae':
                        z_proj = cvae(data)  # This will get z_proj directly using cvae.forward
                    else:
                        z_proj = data  # using original data

                    # Classifier predictions using CVAE representations
                    outputs = classifier(z_proj)
                    val_loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                val_loss /= len(val_data_loader)
                accuracy = 100 * correct / total
                print(f"Epoch {epoch}, Validation Loss: {val_loss}, Accuracy: {accuracy}%")
                return val_loss
         

    if args.train:
        train_losses = []
        val_losses = []
        for epoch in range(1, args.epochs+1):
            print(f'EPOCH {epoch}')
            avg_train_loss = train_one_epoch(epoch)
            train_losses.append(avg_train_loss)

            avg_val_loss = val_one_epoch(epoch)
            val_losses.append(avg_val_loss)

            print(f"Train/Val Sim Loss after epoch: {avg_train_loss:.4f}/{avg_val_loss:.4f}")

    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')

    # add grid
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))  # Set major ticks for every 5 epochs
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.5))  # Set major ticks for every 0.5 on y-axis
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.legend()

    # save loss as different files based on job id
    id = os.getenv('SLURM_JOB_ID')
    if id is None:
        id = 'default'
    
    output_path = f'output/classifier_loss_{id}.pdf'
    plt.savefig(output_path)


if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()

    # If not using full data, name of smaller dataset to pull from
    parser.add_argument('background_dataset', type=str)
    parser.add_argument('anomaly_dataset', type=str)

    parser.add_argument('--mode', type=str, default='original')

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--loss-temp', type=float, default=0.07)
    parser.add_argument('--model-name', type=str, default='output/vae.pth')
    parser.add_argument('--scaling-filename', type=str)
    parser.add_argument('--output-filename', type=str, default='output/embedding.npz')
    parser.add_argument('--sample-size', type=int, default=-1)
    parser.add_argument('--mix-in-anomalies', action='store_true')
    parser.add_argument('--train', action='store_true')

    args = parser.parse_args()
    main(args)

