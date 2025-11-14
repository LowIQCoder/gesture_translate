import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class GestureTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 1001,
        d_model: int = 84,
        d_ff: int = 256,
        num_encoders: int = 3,
        nheads: int = 4, 
        dropout: float = 0.3 
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(84, d_model),
            nn.LayerNorm(d_model)
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer encoder only
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nheads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoders
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.LayerNorm(d_model//2),
            nn.Linear(d_model//2, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Input projection
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, d_model)
        
        # Transformer encoder
        encoded = self.transformer(x)  # (batch, seq_len+1, d_model)
        
        # Use CLS token for classification
        cls_output = encoded[:, 0, :]  # (batch, d_model)
        
        # Classification
        logits = self.classifier(cls_output)  # (batch, num_classes)
        return logits

if __name__ == "__main__":
    model = GestureTransformer()

    exaple_input = torch.rand((1, 1, 84))
    logits = model(exaple_input)

    print(f"Logits shape:\t{logits.shape}")
