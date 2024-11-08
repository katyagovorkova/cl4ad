import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from train import BackgroundDataset, SignalDataset
from model import TransformerModel

import wandb


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(MLPClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.batch_norm(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        return self.layer3(x)


class TfClassifier(nn.Module):
    # takes in original data points as inputs and outpus if prediction was correct
    def __init__(self, input_dim, num_heads, num_classes, latent_dim=3, num_layers=1,\
                  forward_expansion=4, dropout_rate=0.1, batch_first=True, embedding_only=False):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Linear(input_dim, latent_dim)  # Simple embedding layer
        self.pos_embedding = nn.Parameter(torch.zeros(19, latent_dim))  # Positional embeddings
        self.return_embed = embedding_only  # output as classified labels or latent space embeddings
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * forward_expansion,
            dropout=dropout_rate,
            batch_first=batch_first
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.classifier = nn.Linear(latent_dim, num_classes)
    
    def forward(self, x):
        # x shape: (batch, seq, feature)
        batch_size, seq_len, feature_dim = x.shape
        x = self.embedding(x)
        pos = self.pos_embedding[:seq_len, :] # Get the relevant positional embeddings
        x = x + pos.unsqueeze(0).expand(batch_size, -1, -1)  # Add positional embeddings to input embeddings
        x = self.transformer(x)
        x = x.mean(dim=1)  # Average pooling over the sequence
        if self.return_embed:
            return x  # Return the embeddings if self.rep is True
        else:
            out = self.classifier(x)  # Otherwise, apply the classifier
            return out


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

    if args.proportioned=='true':
        data_loader = DataLoader(
            BackgroundDataset(
                dataset['x_test'],
                dataset['ix_test'],
                dataset['labels_test'],
                device),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True)
        val_loader = DataLoader(
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
        x_val = torch.tensor(data['x_test'], dtype=torch.float32).to(device)
        labels_train = torch.tensor(labels['background_ID_train'], dtype=torch.long).to(device)  # Convert labels to tensor
        labels_val = torch.tensor(labels['background_ID_test'], dtype=torch.long).to(device)
        train_dataset = TensorDataset(x_train, labels_train)
        val_dataset = TensorDataset(x_val, labels_val)

        data_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True)
    
    include_anomaly = True if args.include_anomaly == "true" else False

    # Parameters for transformer
    input_dim = 3  # Each step has 3 features
    num_heads = args.heads  # Number of heads in the multi-head attention mechanism
    num_classes = 4  # You have four classes
    num_layers = args.layers  # Number of transformer blocks
    latent_dim = args.latent_dim
    forward_expansion = args.expansion
    dropout_rate = args.dropout
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size  
    embedding_space = True if args.embedding == "true" else False

    # Define the MLP
    input_size = 4  # transformer latent dim
    output_size = 2  # For binary classification: correct or incorrect
    hidden_size = 20
    mlp = MLPClassifier(input_size, hidden_size, output_size).to(device)

    # define tf confidence model
    tf = TransformerModel(3, 4, 2, 4, 4,\
                              2, 0.1, embedding_only=False).to(device)

    wandb.init(
    # set the wandb project where this run will be logged
    project="confidence",
    name=f'{id}',

    # track hyperparameters and run metadata
    config={
        "hidden_size": hidden_size,
        "learning_rate": args.lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "embedding_space": embedding_space,
        "include_anomaly": include_anomaly,
        "notes": args.notes
    }
    )

    model = TransformerModel(input_dim, num_heads, num_classes, latent_dim, num_layers,\
                            forward_expansion, dropout_rate, embedding_only=embedding_space).to(device)

    # load the saved model from the input dir path
    model.load_state_dict(torch.load(args.saved_model))
    model.eval()

    features = []
    val_features = []
    preds = []  # predicted labels
    labels = []  # corresponding labels for each sample
    val_preds = []
    val_labels = []
    samples = 0

    # get the latent space representations
    with torch.no_grad():
        for inputs, label in data_loader:
            inputs = inputs.squeeze(-1)
            samples += inputs.size(0)
            outputs = model(inputs)

            if embedding_space:
                # using transformer outputs as features: (shape=transformer latent dim)
                features.extend(outputs.cpu().detach().numpy())
            else: 
                # using original inputs as features for confidence
                features.extend(inputs.cpu().detach().numpy())

            _, pred = torch.max(outputs, 1)
            preds.extend(pred.cpu().detach().numpy())
            labels.extend(label.cpu().detach().numpy().astype(int))

        for inputs, label in val_loader:
            inputs = inputs.squeeze(-1)
            samples += inputs.size(0)
            outputs = model(inputs)

            if embedding_space:
                # using transformer outputs as features: (shape=transformer latent dim)
                val_features.extend(outputs.cpu().detach().numpy())
            else:
                # using original inputs as features for confidence:
                val_features.extend(inputs.cpu().detach().numpy())

            _, pred = torch.max(outputs, 1)
            
            print('val pred:', pred)
            val_preds.extend(pred.tolist())
            print('val label:', label)
            val_labels.extend(label.tolist())

    if include_anomaly:
        anomaly = np.load(args.anomaly_dataset)
        types = ['leptoquark', 'ato4l', 'hChToTauNu', 'hToTauTau']
        anomaly_data_loader = DataLoader(SignalDataset(anomaly, types, device), batch_size=batch_size)

        ad_features = []
        ad_preds = []
        ad_labels = []

        # get latent space representations
        with torch.no_grad():
            for inputs, label in anomaly_data_loader:
                inputs = inputs.squeeze(-1)
                outputs = model(inputs)
                samples += inputs.size(0)

                if embedding_space:
                    # using transformer outputs as features: (shape=transformer latent dim)
                    ad_features.extend(outputs.cpu().detach().numpy())
                else:
                    # using original inputs as features for confidence:
                    ad_features.extend(inputs.cpu().detach().numpy())

                _, pred = torch.max(outputs, 1)
                
                ad_preds.extend(pred.cpu().detach().numpy())
                ad_labels.extend(label.cpu().detach().numpy().astype(int))

    # preds = np.concatenate(preds, axis=0)
    # labels = np.concatenate(labels, axis=0)
    # val_preds = np.concatenate(val_preds, axis=0)
    # val_labels = np.concatenate(val_labels, axis=0)
    preds = np.array(preds)
    labels = np.array(labels)
    features = np.array(features)
    val_features = np.array(val_features)
    val_preds = np.array(val_preds)
    val_labels = np.array(val_labels)
    # print('Number of samples:', samples)
    # print('Number of labels:', labels.shape)
    # print(preds.shape)
    # print(features.shape)
    # print(val_features.shape)
    # print()

    # Prepare input features for confidence model
    # using original inputs & tagging on predictions at the end
    # features = np.hstack((features.reshape(features.shape[0], -1), preds[:, None]))  # Adding prediction as a feature
    # using transformer outputs, no predictions: no change,  features=features
    
    features = torch.tensor(features, dtype=torch.float32).to(device)
    correctness = (preds == labels).astype(int)
    y = torch.tensor(correctness, dtype=torch.int64).to(device)

    # Create a dataset and dataloader for training
    dataset = TensorDataset(features, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # testing
    # using original inputs & tagging on predictions at the end
    # test_features = np.hstack((val_features.reshape(val_features.shape[0], -1), val_preds[:, None]))
    # using transformer outputs, no predictions: no change,  features=features
    test_features = torch.tensor(val_features, dtype=torch.float32).to(device)
    test_correctness = (val_preds == val_labels).astype(int)
    test_y = torch.tensor(test_correctness, dtype=torch.int64).to(device)
    test_dataset = TensorDataset(test_features, test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.AdamW(mlp.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    # anomaly detection
    if include_anomaly:
        # only used in validation
        ad_features = np.array(ad_features)
        ad_preds = np.array(ad_preds)
        ad_labels = np.array(ad_labels)

        ad_features = torch.tensor(ad_features, dtype=torch.float32).to(device)
        ad_correctness = (ad_preds == ad_labels).astype(int)  # should be all false
        print('ad correctness:', ad_correctness)
        ad_y = torch.tensor(ad_correctness, dtype=torch.int64).to(device)
        ad_dataset = TensorDataset(ad_features, ad_y)
        ad_loader = DataLoader(ad_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    def train_epoch(dataloader):
        model.train()
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            if embedding_space:
                outputs = mlp(inputs)
            else:
                inputs = inputs.squeeze(-1)  # shape: (batch, seq_len, input_dim)
                outputs = tf(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        return loss.item()

    def val_epoch(dataloader):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                if embedding_space:
                    outputs = mlp(inputs)
                else:
                    inputs = inputs.squeeze(-1)  # shape: (batch, seq_len, input_dim)
                    outputs = tf(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                mislabels = (predicted == 0).sum().item()
                print("number of 1's test:", (predicted == 1).sum().item())
        return loss.item(), correct, total, mislabels

    def ad_epoch(dataloader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                if embedding_space:
                    outputs = mlp(inputs)
                else:
                    inputs = inputs.squeeze(-1)  # shape: (batch, seq_len, input_dim)
                    outputs = tf(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                mislabels = (predicted == 0).sum().item()
                # print("number of 1's anomaly:", (predicted == 1).sum().item())
        return correct, total, mislabels

    for epoch in range(epochs):
        train_loss = train_epoch(train_loader)
        test_loss, correct, total, mislabels = val_epoch(test_loader)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}")
        print(f'Accuracy of the network on the test events: {correct / total}')
        print(f'Test loss: {test_loss}')
        print(f'Predicted mislabels in test: {mislabels}')
        if include_anomaly:
            ad_correct, ad_total, ad_mislabels = ad_epoch(ad_loader)
            wandb.log({"anomaly accuracy": ad_correct/ad_total, \
                       "correctly predicted mislabels": ad_mislabels/(mislabels + ad_mislabels)})
            print(f'Accuracy of anomaly: {ad_correct / ad_total}')
            print(f'Predicted mislabels in anomaly: {ad_mislabels}')
        print()
        scheduler.step()
        wandb.log({"train loss": train_loss, "test loss": test_loss, "test accuracy": correct / total})

    wandb.finish()




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
    parser.add_argument('--embedding', type=str)

    parser.add_argument('--output-dir', type=str, default='output/tf_cluster/')
    parser.add_argument('--proportioned', type=str, default=None)

    parser.add_argument('--latent-dim', type=int, default=3)
    parser.add_argument('--heads', type=int, default=3)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--expansion', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--notes', type=str)

    args = parser.parse_args()
    main(args)
