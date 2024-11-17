import numpy as np
from argparse import ArgumentParser
import os

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from model import TransformerModel

torch.autograd.set_detect_anomaly(True)


id = os.getenv('SLURM_JOB_ID')
if id is None:
    id = 'default'

class BackgroundDataset(Dataset):
    'Characterizes the background dataset for PyTorch'
    def __init__(self, x, ix, labels, device, augmentation=False, ixa=None):
          'Initialization'
          self.device = device
          self.data = torch.from_numpy(x[ix]).to(dtype=torch.float32, device=self.device)
          self.labels = torch.from_numpy(labels[ix]).to(dtype=torch.long, device=self.device)
          # if augmentation, prepare augmented outputs for vicreg loss
        #   ixa = np.concatenate((ix[1:], ix[0:1]))
          self.augmented_data = torch.from_numpy(x[ixa]).to(dtype=torch.float32, device=self.device) if augmentation else None

    def __len__(self):
          'Denotes the total number of samples'
          return len(self.data)

    def __getitem__(self, index):
          'Generates one sample of data'
          if self.augmented_data is not None:
              return self.data[index], self.augmented_data[index], self.labels[index]
          return self.data[index], self.labels[index]


class SignalDataset(Dataset):
    """ Prepares an anomaly dataset for PyTorch """
    def __init__(self, data_dict, types, device):
        """ 
        assumes data_dict is the dictionary containing data and labels for anomalies
        types: list of types to include in the dataset, e.g. ['leptoquark', 'ato4l', 'hChToTauNu', 'hToTauTau']
        """
        self.data = []
        self.labels = []
        self.device = device

        # Iterate over specified types and append data and labels to lists
        for t in types:
            data_key = t
            label_key = f'labels_{t}'
            labels = data_dict[label_key].copy()
            # Convert numpy arrays to torch tensors and append to lists
            self.data.append(torch.from_numpy(data_dict[data_key]).to(dtype=torch.float32, device = self.device))
            self.labels.append(torch.from_numpy(labels.reshape((labels.shape[0],)).astype(int)).to(dtype=torch.long, device = self.device))
        
        # Concatenate all tensors from list into a single tensor
        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, idx):
        'Generates one sample of data'
        return self.data[idx], self.labels[idx]
    


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    print(args.notes)
    print(args.latent_dim)
    print(args.model_dir)

    torch.manual_seed(0)
    np.random.seed(0)

    # Parameters
    input_dim = 3  # Each step has 3 features
    num_heads = args.heads  # Number of heads in the multi-head attention mechanism
    num_classes = 4  # You have four classes
    num_layers = args.layers  # Number of transformer blocks
    latent_dim = args.latent_dim
    forward_expansion = args.expansion
    dropout_rate = args.dropout
    save_embed = True if args.save_embed == "true" else False

    model = TransformerModel(input_dim, num_heads, num_classes, latent_dim, num_layers, forward_expansion, dropout_rate).to(device)
    model.load_state_dict(torch.load(f"output/transformer/transformer_{args.model_dir}.pth"))

    dataset = np.load(args.background_dataset)
    test_data_loader = DataLoader(
        BackgroundDataset(
            dataset['x_test'],
            dataset['ix_test'],
            dataset['labels_test'],
            device,
            ),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        )

    instances = []
    embeds = []
    preds = []
    labels = []

    for data, label in test_data_loader:
        with torch.no_grad():
            inputs = data.squeeze(-1)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)

            instances.append(inputs.to(device).tolist())
            embeds.append(outputs.to(device).tolist())
            preds.extend(pred.to(device).tolist())
            labels.extend(label.to(device).tolist())

    correct_predictions = sum([1 for pred, true in zip(preds, labels) if pred == true])
    accuracy = correct_predictions / len(preds)

    instances = np.array(instances)
    embeds = np.array(embeds)
    preds = np.array(preds)
    labels = np.array(labels)

    preds_dir = args.preds_dir
    if preds_dir is not None:
        with np.load(preds_dir) as data:
            data_dict = {key: data[key] for key in data.keys()}
        
        data_dict[f'dim{latent_dim}_embeddings'] = embeds
        data_dict[f'dim{latent_dim}_predictions'] = preds
        data_dict[f'dim{latent_dim}_labels'] = labels
        data_dict[f'dim{latent_dim}_accuracy'] = accuracy
        np.savez(preds_dir, **data_dict)

    else:
        preds_dir = f"vicreg_predictions.npz"
        data_dict = {'instances': instances, f'dim{latent_dim}_embeddings': embeds, f'dim{latent_dim}_predictions': preds, \
                      f'dim{latent_dim}_labels': labels, f'dim{latent_dim}_accuracy': accuracy}
        np.savez(preds_dir, **data_dict)


if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()

    parser.add_argument('--data-filename', type=str, default=None)
    parser.add_argument('--labels-filename', type=str, default=None)
    parser.add_argument('--background-dataset', type=str, default=None)
    parser.add_argument('--kfold-dataset', type=str, default=None)
    parser.add_argument('--anomaly-dataset', type=str)

    parser.add_argument('--latent-dim', type=int, default=3)
    parser.add_argument('--heads', type=int, default=3)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--expansion', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--model-dir', type=int)
    parser.add_argument('--preds-dir', type=str, default=None)
    parser.add_argument('--save-embed', type=str, default="true")

    parser.add_argument('--notes', type=str)

    args = parser.parse_args()
    main(args)