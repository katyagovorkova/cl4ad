import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
torch.autograd.set_detect_anomaly(True)


import losses
from models import CVAE

id = os.getenv('SLURM_JOB_ID')
if id is None:
    id = 'default'

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
    '''
    Infastructure for training CVAE (background specific and with anomalies)
    '''

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    # background dataset
    dataset = np.load(args.background_dataset)
    visualize = args.visualize

    # criterion = losses.SimCLRLoss()
    criterion = losses.VICRegLoss()

    # load the datasets for pytorch
    train_data_loader = DataLoader(
        TorchCLDataset(
            dataset['x_train'],
            dataset['ix_train'],
            dataset['ixa_train'],
            dataset['labels_train'],
            criterion, device),
        batch_size=args.batch_size,
        shuffle=False)

    test_data_loader = DataLoader(
        TorchCLDataset(
            dataset['x_test'],
            dataset['ix_test'],
            dataset['ixa_test'],
            dataset['labels_test'],
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

    model = CVAE().to(device)
    summary(model, input_size=(57,))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler_1 = torch.optim.lr_scheduler.ConstantLR(optimizer, total_iters=5)
    scheduler_2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler_3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler_1, scheduler_2, scheduler_3], milestones=[5, 20])


    # training
    def train_one_epoch(epoch_index):
        running_sim_loss = 0.
        last_sim_loss = 0.

        for idx, (val, val_aug, _) in enumerate(train_data_loader, 1):
            # only applicable to the final batch
            if val.shape[0] != args.batch_size:
                continue

            embedded_values_orig = model(val)
            embedded_values_aug = model(val_aug)

            similar_embedding_loss = criterion(embedded_values_aug.reshape((-1,6)), \
                embedded_values_orig.reshape((-1,6)))

            optimizer.zero_grad()
            similar_embedding_loss.backward()
            optimizer.step()
            # Gather data and report
            running_sim_loss += similar_embedding_loss.item()
            if idx % 500 == 0:
                last_sim_loss = running_sim_loss / 500
                running_sim_loss = 0.

        return last_sim_loss


    # validation 
    def val_one_epoch(epoch_index):
        running_sim_loss = 0.
        last_sim_loss = 0.

        for idx,(val, val_aug, _) in enumerate(val_data_loader, 1):

            if val.shape[0] != args.batch_size:
                continue

            embedded_values_orig = model(val)
            embedded_values_aug = model(val_aug)

            similar_embedding_loss = criterion(embedded_values_aug.reshape((-1,6)), \
                embedded_values_orig.reshape((-1,6)))

            running_sim_loss += similar_embedding_loss.item()
            if idx % 500 == 0:
                last_sim_loss = running_sim_loss / 500
                running_sim_loss = 0.

        return last_sim_loss
    
    # early stopping
    class EarlyStopping:
        def __init__(self, save_path, patience=10, verbose=False, delta=0):
            """
            Args:
                save_path (str): Path to save the checkpoint model.
                patience (int): How long to wait after last time validation loss improved.
                                Default: 10
                verbose (bool): If True, prints a message for each validation loss improvement. 
                                Default: False
                delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            """
            self.save_path = save_path
            self.patience = patience
            self.verbose = verbose
            self.delta = delta
            self.best_loss = float('inf')
            self.epochs_no_improve = 0
            self.should_stop = False

        def check(self, val_loss, model):
            if val_loss < self.best_loss - self.delta:
                if self.verbose:
                    print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
                torch.save(model.state_dict(), self.save_path)
                self.best_loss = val_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    self.should_stop = True

    early_stopping = EarlyStopping(save_path=args.model_name, patience=10, verbose=True)

    if args.train:
        train_losses = []
        val_losses = []
        for epoch in range(1, args.epochs+1):
            print(f'EPOCH {epoch}')
            # Gradient tracking
            model.train(True)
            avg_train_loss = train_one_epoch(epoch)
            train_losses.append(avg_train_loss)

            # no gradient tracking, for validation
            # model.train(False)
            model.eval()
            avg_val_loss = val_one_epoch(epoch)
            val_losses.append(avg_val_loss)

            print(f"Train/Val Sim Loss after epoch: {avg_train_loss:.4f}/{avg_val_loss:.4f}")

            scheduler.step()

            early_stopping.check(avg_val_loss, model)
            if early_stopping.should_stop:
                print(f"Stopping early at epoch {epoch}")
                break

        torch.save(model.state_dict(), f'output/vae_{id}.pth')
    else:
        model.load_state_dict(torch.load(args.model_name))
        model.eval()


    # plot the loss curve

    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')

    # set the y range
    plt.ylim([12, 20])

    # add grid
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))  # Set major ticks for every 10 epochs
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))  # Set major ticks for every 1 on y-axis
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.legend()

    # save loss as different files based on job id
    
    output_path = f'output/loss_{id}.pdf'
    plt.savefig(output_path)
    # plt.savefig('output/loss.pdf')


    # plot the visualization of latent space on validation set


if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()

    # If not using full data, name of smaller dataset to pull from
    parser.add_argument('background_dataset', type=str)
    parser.add_argument('anomaly_dataset', type=str)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--loss-temp', type=float, default=0.07)
    parser.add_argument('--model-name', type=str, default='output/vae.pth')
    parser.add_argument('--scaling-filename', type=str)
    parser.add_argument('--output-filename', type=str, default='output/embedding.npz')
    parser.add_argument('--sample-size', type=int, default=-1)
    parser.add_argument('--mix-in-anomalies', action='store_true')
    parser.add_argument('--train', action='store_true')

    parser.add_argument('--visualize', type=bool, default=False)
    parser.add_argument('--visualized_plots', type=str)

    args = parser.parse_args()
    main(args)
