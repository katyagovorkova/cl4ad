import torch
import torch.nn as nn

class TransformerModel(nn.Module):
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

