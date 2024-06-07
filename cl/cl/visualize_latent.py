import os
import numpy as np
from argparse import ArgumentParser

import torch
from torchsummary import summary

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
    data = dataset['x_val']
    labels = dataset['labels_val']

    # load the saved model from the input dir path
    model = CVAE().to(device)
    summary(model, input_size=(57,))

    model.load_state_dict(torch.load(args.saved_model))
    model.eval()


    # get the latent space representations
    latent_z = model.representation(data)


    # Pairwise scatter plots
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


    # t-SNE plots
    # Reduce dimensions to 2 using t-SNE
    tsne = TSNE(n_components=2)
    latent_2d_tsne = tsne.fit_transform(latent_z)

    # Plotting the 2D t-SNE result with color coding by label
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d_tsne[:, 0], latent_2d_tsne[:, 1], c=labels, cmap='viridis')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE of Latent Space (Validation Data)')
    plt.colorbar(scatter)
    plt.savefig(os.path.join(save_dir, f'tsne_plot_{id}.png'))
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