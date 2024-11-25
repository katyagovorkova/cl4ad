import numpy as np
from argparse import ArgumentParser
import os
import re

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

    torch.manual_seed(0)
    np.random.seed(0)

    dataset = np.load(args.background_dataset)
    dataset_part = args.dataset_part  # test or train or val
    signal_dataset = np.load(args.anomaly_dataset)

    data_loader = DataLoader(
                    BackgroundDataset(
                        dataset[f'x_{dataset_part}'],
                        dataset[f'ix_{dataset_part}'],
                        dataset[f'labels_{dataset_part}'],
                        device=device,
                        augmentation=True, # change to simclr when needed
                        ixa=dataset[f'ixa_{dataset_part}'],),
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last=True)
        
    signal_data_loader = DataLoader(
        SignalDataset(
            signal_dataset,
            ['leptoquark', 'ato4l', 'hChToTauNu', 'hToTauTau'],
            device=device
        ),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True
    )

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
        loss_name = "vicreg" if vicreg else "simclr"

        dim = int(match.group(1)) 
        heads = int(match.group(2))  
        fold = int(match.group(3))
        layers = 4
        expansion = 16
        print(f"dim: {dim}, head: {heads}, fold: {fold}")

        model = TransformerModel(3, heads, 4, dim, layers, expansion, dropout_rate=0.1, embedding_only=True).to(device)
        model.load_state_dict(torch.load(model_prefix + suffix))
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        fold_instances = []
        fold_embeds = []
        fold_preds = []
        fold_labels = []

        if vicreg:
            for x, _, label in data_loader:  #_ is augmented value
                with torch.no_grad():
                    x = x.squeeze(-1)
                    outputs = model(x)
                    _, pred = torch.max(outputs, 1)

                    fold_instances.append(x.to(device).tolist())
                    fold_embeds.append(outputs.to(device).tolist())
                    fold_preds.extend(pred.to(device).tolist())
                    fold_labels.extend(label.to(device).tolist())
        else:
            for x, label in data_loader:
                with torch.no_grad():
                    x = x.squeeze(-1)
                    outputs = model(x)
                    _, pred = torch.max(outputs, 1)

                    fold_instances.append(x.to(device).tolist())
                    fold_embeds.append(outputs.to(device).tolist())
                    fold_preds.extend(pred.to(device).tolist())
                    fold_labels.extend(label.to(device).tolist())
        for x, label in signal_data_loader:
            with torch.no_grad():
                x = x.squeeze(-1)
                outputs = model(x)
                _, pred = torch.max(outputs, 1)

                fold_instances.append(x.to(device).tolist())
                fold_embeds.append(outputs.to(device).tolist())
                fold_preds.extend(pred.to(device).tolist())
                fold_labels.extend(label.to(device).tolist())

        # correct_predictions += sum([1 for pred, true in zip(fold_preds, fold_labels) if pred == true])
        # num_samples = len(data_loader.dataset) + len(signal_data_loader.dataset)
        fold_instances = np.array(fold_instances)
        print(fold_instances.shape)
        fold_instances = fold_instances.reshape(1, -1, fold_instances.shape[-2], fold_instances.shape[-1])
        print(fold_instances.shape)
        fold_embeds = np.array(fold_embeds)
        print(fold_embeds.shape)
        fold_embeds = fold_embeds.reshape(1, -1, 32)
        print(fold_embeds.shape)
        fold_preds = np.array(fold_preds).reshape(1,-1)
        print(fold_preds.shape)
        fold_labels = np.array(fold_labels).reshape(1,-1)
        print(fold_labels.shape)
        if fold == 0:
            instances = fold_instances.copy()
            embeds = fold_embeds.copy()
            preds = fold_preds.copy()
            labels = fold_labels.copy()
        else:
            # fold_instances = np.array(fold_instances).reshape(-1,1)
            # fold_embeds = np.array(fold_embeds).reshape(-1,1)
            # fold_preds = np.array(fold_preds).reshape(-1,1)
            # fold_labels = np.array(fold_labels).reshape(-1,1)

            print('instances ',instances.shape)
            instances = np.concatenate((instances, fold_instances), axis=0)
            print('embeds', embeds.shape)
            embeds = np.concatenate((embeds, fold_embeds), axis=0)
            print('preds', preds.shape)
            preds = np.concatenate((preds, fold_preds),axis=0)
            print('labels', labels.shape)
            labels = np.concatenate((labels, fold_labels),axis=0)

    preds_dir = args.preds_dir
    if preds_dir is not None:
        with np.load(preds_dir) as data:
            data_dict = {key: data[key] for key in data.keys()}
        
        data_dict[f'dim{dim}_instances'] = instances
        data_dict[f'dim{dim}_embeddings'] = embeds
        data_dict[f'dim{dim}_predictions'] = preds
        data_dict[f'dim{dim}_labels'] = labels
        np.savez(preds_dir, **data_dict)

    else:
        preds_dir = f"{loss_name}_dim{dim}_kfold_predictions.npz"
        data_dict = {f'dim{dim}_instances': instances, f'dim{dim}_embeddings': embeds, f'dim{dim}_predictions': preds, \
                    f'dim{dim}_labels': labels}
        np.savez(preds_dir, **data_dict)



if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()

    parser.add_argument('--background-dataset', type=str, default=None)
    parser.add_argument('--anomaly-dataset', type=str, default=None)
        # e.g. output/anomaly_dataset.npz
    parser.add_argument('--dataset-part', type=str, default=None)
        # e.g. test
    parser.add_argument('--kfold-dataset', type=str, default=None)
    # parser.add_argument('--layers', type=int, default=1)
    # parser.add_argument('--expansion', type=int, default=4)
    # parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--preds-dir', type=str, default=None)
    parser.add_argument('--model-dir', type=str, default=None)
        # e.g. /n/home12/ylanaxu/orca/output/kfold_exp/
    parser.add_argument('--model-names', type=str, default=None)
        # e.g. vicreg_dim2_head2_56194334,vicreg_dim4_head4_56194342,vicreg_dim8_head4_56194352,vicreg_dim16_head8_56194369,vicreg_dim24_head8_56194372,vicreg_dim32_head16_56194387

    parser.add_argument('--notes', type=str)

    args = parser.parse_args()
    main(args)