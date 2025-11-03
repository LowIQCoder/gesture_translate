import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, emb_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq, emb)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


class LandmarkTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_landmarks=84,
        num_classes=1001,
        emb_dim=128,
        num_heads=4,
        num_encoders=4,
        num_decoders=5,
        dim_ff=512,
        dropout=0.1,
        max_seq_len=256,
        pooling='mean'
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.pooling = pooling

        # Map landmarks -> embedding dimension
        self.input_proj = nn.Linear(num_landmarks, emb_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(emb_dim, max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoders)

        # Learnable class token (acts as decoder query)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, num_landmarks)
        mask: optional BoolTensor (batch, seq_len) where True = pad token to ignore
        """
        b, seq, _ = x.shape
        x = self.input_proj(x)         # (b, seq, emb)
        x = self.pos_encoder(x)
        memory = self.encoder(x, src_key_padding_mask=mask)  # (b, seq, emb)
    
        cls_query = self.cls_token.expand(b, -1, -1)  # (b, 1, emb)
        tgt = cls_query

        # Decode
        decoded = self.decoder(tgt=tgt, memory=memory, memory_key_padding_mask=mask)  # (b, 1, emb)
        decoded = decoded.squeeze(1)  # (b, emb)

        out = self.dropout(decoded)
        logits = self.fc(out)  # (b, num_classes)
        return logits


if __name__ == "__main__":
    batch = 8
    seq_len = 140
    num_landmarks = 84
    num_classes = 1000

    model = LandmarkTransformerClassifier(
        num_landmarks=num_landmarks,
        num_classes=num_classes,
        emb_dim=128,
        num_heads=4,
        num_layers=4,
        pooling='mean'
    )

    # Dummy data
    x = torch.randn(batch, seq_len, num_landmarks) # no padding
    logits = model(x)  # (batch, num_classes)
    print("Logits:", logits.shape)  # (8, 400)
