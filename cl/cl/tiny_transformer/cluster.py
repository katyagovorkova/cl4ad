import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset, TensorDataset

from train import BackgroundDataset, SignalDataset
from model import TransformerModel

NAME_MAPPINGS = {
    0:'W-Boson',
    1:'QCD',
    2:'Z_2',
    3:'tt',
    4:'leptoquark',
    5:'ato4l',
    6:'hChToTauNu',
    7:'hToTauTau'
}


id = os.getenv('SLURM_JOB_ID')
if id is None:
    id = 'default'


def main(args):
    save_dir = args.output_dir
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    print(args.notes)

    # load the background dataset 
    dataset = np.load(args.background_dataset)

    if args.proportioned:
        # proportioned dataset
        dataset = np.load(args.background_dataset)
        data_loader = DataLoader(
            BackgroundDataset(
                dataset['x_val'],
                dataset['ix_val'],
                dataset['labels_val'],
                device),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True)
    else:
        # raw background dataset
        data = np.load(args.data_filename, mmap_mode='r')
        labels = np.load(args.labels_filename, mmap_mode='r')
        x_train = torch.tensor(data['x_train'], dtype=torch.float32).to(device)  # Convert data to tensor
        x_val = torch.tensor(data['x_val'], dtype=torch.float32).to(device)
        labels_train = torch.tensor(labels['background_ID_train'], dtype=torch.long).to(device)  # Convert labels to tensor
        labels_val = torch.tensor(labels['background_ID_val'], dtype=torch.long).to(device)
        train_dataset = TensorDataset(x_train, labels_train)
        val_dataset = TensorDataset(x_val, labels_val)

        data_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True)
    
    include_anomaly = True if args.include_anomaly == "true" else False

    # Parameters
    input_dim = 3  # Each step has 3 features
    num_heads = args.heads  # Number of heads in the multi-head attention mechanism
    num_classes = 4  # You have four classes
    num_layers = args.layers  # Number of transformer blocks
    latent_dim = args.latent_dim
    forward_expansion = args.expansion
    dropout_rate = args.dropout

    model = TransformerModel(input_dim, num_heads, num_classes, latent_dim, num_layers,\
                              forward_expansion, dropout_rate, embedding_only=True).to(device)

    # load the saved model from the input dir path
    model.load_state_dict(torch.load(args.saved_model))
    model.eval()

    latent_z = []  # holds all the latent embeddings
    labels = []  # corresponding labels for each sample
    samples = 0

    # get the latent space representations
    for inputs, label in data_loader:
        inputs = inputs.squeeze(-1)
        embedding = model(inputs).cpu().detach().numpy()
        samples += inputs.size(0)
        latent_z.append(embedding)
        labels.append(label.cpu().detach().numpy().astype(int))

    if include_anomaly:
        anomaly = np.load(args.anomaly_dataset)
        types = ['leptoquark', 'ato4l', 'hChToTauNu', 'hToTauTau']
        anomaly_data_loader = DataLoader(SignalDataset(anomaly, types, device), batch_size=args.batch_size)

        # get latent space representations
        for inputs, label in anomaly_data_loader:
            print('anomaly input shape:', inputs.shape)
            inputs = inputs.squeeze(-1)
            print('after squeeze:', inputs.shape)
            embedding = model(inputs).cpu().detach().numpy()
            samples += inputs.size(0)
            latent_z.append(embedding)
            labels.append(label.cpu().detach().numpy().astype(int))

    latent_z = np.concatenate(latent_z, axis=0)
    labels = np.concatenate(labels, axis=0)
    print('Number of samples:', samples)
    print('Number of labels:', labels.shape)
    print()


    print('Making plots')
    # Pairwise scatter plots between latent space dimensions
    # Convert the latent space to a DataFrame
    dsp = pd.DataFrame(latent_z, columns=[f'Dim{i}' for i in range(1, latent_dim+1)])
    dsp['label'] = labels

    # Pairwise scatter plot with color coding by label
    pairplot = sns.pairplot(dsp, hue='label', palette='viridis', diag_kind='hist', diag_kws=dict(stat='density'))
    pairplot.savefig(os.path.join(save_dir, f'pairwise_scatter_{id}.png'))
    plt.close()

    # Plotting only the diagonal histograms
    num_dims = latent_dim
    fig, axes = plt.subplots(nrows=1, ncols=num_dims, figsize=(15, 3))
    palette = sns.color_palette("viridis", len(dsp['label'].unique()))
    # Loop through each dimension and plot a histogram for each label
    handles = []  # To store handles for the legend
    labels_list = []  # To store labels for the legend
    for i in range(num_dims):
        ax = axes[i]  # Current axis
        for label, color in zip(sorted(dsp['label'].unique()), palette):
                # Subset data for the current label
                subset = dsp[dsp['label'] == label][f'Dim{i+1}']
                # Plot histogram and store the handle
                n, bins, patches = ax.hist(subset, bins=20, density=True, alpha=0.7, color=color, label=f'{NAME_MAPPINGS[label]}')
                if i == 0:  # Only add handles and labels once
                    handles.append(patches[0])
                    labels_list.append(f'{NAME_MAPPINGS[label]}')
        ax.set_title(f'Dim{i+1} Histogram')
        ax.set_xlabel(f'Dim{i+1}')
        ax.set_ylabel('Density')

    # TODO: fix why is this not working 37790254
    # fig, axes = plt.subplots(nrows=len(types) + 1, ncols=num_dims, figsize=(15, 3)) if include_anomaly \
    #     else plt.subplots(1, ncols=num_dims, figsize=(15, 3))
    # palette = sns.color_palette("viridis", 8)
    # # first only plot the background
    # for i in range(num_dims):
    #     ax = axes[0,i] if include_anomaly else axes[i]  # Current axis

    #     for label in range(4):  # backgrounds are 0-3
    #         subset = dsp[dsp['label'] == label][f'Dim{i+1}']
    #         # Plot histogram and store the handle
    #         n, bins, patches = ax.hist(subset, bins=20, density=True, alpha=0.7, color=palette[label], label=f'{NAME_MAPPINGS[label]}')
    #         # if i == 0:  # Only add handles and labels once
    #         #     handles.append(patches[0])
    #         #     labels_list.append(f'{NAME_MAPPINGS[label]}')
    #     ax.set_title('Background densities')
    #     ax.set_xlabel(f'Dim{i+1}')
    #     ax.set_ylabel('Density')

    # if include_anomaly:
    #     types = ['leptoquark', 'ato4l', 'hChToTauNu', 'hToTauTau']
    #     fig_num = 0
    #     for anomaly in types:
    #         anomaly_num = fig_num + 4
    #         for i in range(num_dims):
    #             ax = axes[fig_num, i]
    #             for label in range(4):  # plot the background data
    #                 subset = dsp[dsp['label'] == label][f'Dim{i+1}']
    #                 n, bins, patches = ax.hist(subset, bins=20, density=True, alpha=0.7, color=palette[label], label=f'{NAME_MAPPINGS[label]}')

    #             anomaly_subset = dsp[dsp['label'] == anomaly_num][f'Dim{i+1}']
    #             n, bins, patches = ax.hist(subset, bins=20, density=True, alpha=0.7, color=palette[anomaly_num], label=anomaly)
    #             ax.set_title(f'Background with {anomaly}')
    #             ax.set_xlabel(f'Dim{i+1}')
    #             ax.set_ylabel('Density')
    #         #     if i==0:
    #         #         handles.append(patches[0][-1])

    #         # labels_list.append(anomaly)
    #         fig_num += 1
    # handles, labels_list = ax.get_legend_handles_labels()

    # Adjust layout to prevent overlap
    fig.tight_layout()

    # Add a legend to the figure outside of the last axes
    
    fig.legend(handles, labels_list)

    # Save the plot
    plt.savefig(os.path.join(save_dir, f'density_histogram_{id}.png'))
    # plt.close()



if __name__ == '__main__':
    #Parses terminal command
    parser = ArgumentParser()

    # inputs for preprocessed background and anomaly datasets (N, 19, 3)
    parser.add_argument('--data-filename', type=str, default=None)  # raw data
    parser.add_argument('--labels-filename', type=str, default=None)
    parser.add_argument('--background-dataset', type=str, default=None)  # proportioned data
    parser.add_argument('--anomaly-dataset', type=str)

    parser.add_argument('--saved-model', type=str)
    parser.add_argument('--include-anomaly', type=str)  # "true" or not

    parser.add_argument('--output-dir', type=str, default='output/tf_cluster/')
    parser.add_argument('--proportioned', type=str, default=None)

    parser.add_argument('--latent-dim', type=int, default=3)
    parser.add_argument('--heads', type=int, default=3)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--expansion', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=256)

    parser.add_argument('--notes', type=str)

    args = parser.parse_args()
    main(args)