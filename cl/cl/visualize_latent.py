import os
import numpy as np
from argparse import ArgumentParser
from train import TorchCLDataset

import torch
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset

from sklearn.decomposition import PCA
import umap

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from models import CVAE

"""
Running this file will create visualizations of the embeddings in the 6-dim latent space of CVAE

Inputs fron the terminal:
    background_dataset
    saved_model
    (optional) anomaly_dataset
    
Outputs:
    output-filename: PDFs of visualizations
"""

id = os.getenv('SLURM_JOB_ID')
if id is None:
    id = 'default'


def main(args):
    save_dir = args.output_dir

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    # load the background dataset 
    dataset = np.load(args.background_dataset)

    # check the proportions of each type of label
    ix_train = dataset['ix_train']
    ix_test = dataset['ix_test']
    ix_val = dataset['ix_val']
    labels_val = dataset['labels_val'][ix_val]
    labels_train = dataset['labels_train'][ix_train]
    labels_test = dataset['labels_test'][ix_test]

    for labels in [labels_train, labels_test, labels_val]:
        # Get unique labels and their counts
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Calculate the proportion of each label
        total_labels = labels.size
        proportions = counts / total_labels

        # Print the results
        for label, proportion in zip(unique_labels, proportions):
            print(f"Label {(int(label))}: {proportion:.2f}")
        print()

    # load the data
    val_data_loader = DataLoader(
        TorchCLDataset(
            dataset['x_val'],
            dataset['ix_val'],
            dataset['ixa_val'],
            dataset['labels_val'],
            device),
        batch_size=512,
        shuffle=False)

    # load the saved model from the input dir path
    model = CVAE().to(device)
    summary(model, input_size=(57,))

    model.load_state_dict(torch.load(args.saved_model))
    model.eval()

    latent_z = []  # holds all the latent representations
    samples = 0
    # get the latent space representations
    for idx,(val, val_aug, _) in enumerate(val_data_loader, 1):
        representation = model.representation(val).cpu().detach().numpy().reshape((-1,6))
        samples += val.size(0)
        latent_z.append(representation)

    latent_z = np.concatenate(latent_z, axis=0)
    print('Number of samples:', samples)



    print('Making plots')
    # Pairwise scatter plots between latent space dimensions
    # Convert the latent space to a DataFrame
    dsp = pd.DataFrame(latent_z, columns=[f'Dim{i}' for i in range(1, 7)])
    dsp['label'] = labels

    # Pairwise scatter plot with color coding by label
    pairplot = sns.pairplot(dsp, hue='label', palette='viridis')
    pairplot.savefig(os.path.join(save_dir, f'pairwise_scatter_{id}.png'))
    plt.close()


    # PCA plots
    # Reduce dimensions to 2 using PCA
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_z)

    # Plotting the 2D PCA result with color coding by label
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='viridis')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA of Latent Space (Validation Data)')
    plt.colorbar(scatter)
    plt.savefig(os.path.join(save_dir, f'pca_plot_{id}.png'))
    plt.close()


    # UMAP plots
    # Reduce dimensions to 2 using UMAP
    umap_model = umap.UMAP(n_components=2)
    latent_2d_umap = umap_model.fit_transform(latent_z)

    # Plotting the 2D UMAP result with color coding by label
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d_umap[:, 0], latent_2d_umap[:, 1], c=labels, cmap='viridis')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.title('UMAP of Latent Space (Validation Data)')
    plt.colorbar(scatter)
    plt.savefig(os.path.join(save_dir, f'umap_plot_{id}.png'))
    plt.close()



if __name__ == '__main__':
    #Parses terminal command
    parser = ArgumentParser()

    # inputs for preprocessed background and anomaly datasets (ones saved from create_dataset.py)
    parser.add_argument('background_dataset', type=str)
    parser.add_argument('saved_model', type=str)

    parser.add_argument('--anomaly-dataset', type=str)
    parser.add_argument('--output-dir', type=str, default='output/visualization_plots')

    args = parser.parse_args()
    main(args)