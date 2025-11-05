import torch
import torch.nn as nn
import torch.nn.functional as F

class GestureCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        d_model: int = 84,
        d_hidden: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()

        self.input_proj = nn.Linear(84, d_model)  
        
        # Convolution layers
        self.convs = nn.Sequential(
            # After input_proj: (batch, seq_len, d_model)  transpose  (batch, d_model, seq_len)
            nn.Conv1d(d_model, d_hidden, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),  # seq_len: 100 → 50
            
            nn.Conv1d(d_hidden, d_hidden * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # seq_len: 50 → 25
            
            nn.Conv1d(d_hidden * 2, d_hidden * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # (batch, d_hidden*4, 1)
        )   
        
        # Classifier - input size must match last conv layer output channels
        self.fc = nn.Linear(d_hidden * 4, num_classes)   
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len=100, features=84)
        x = self.input_proj(x)  # (batch_size, 100, d_model)
        x = x.transpose(1, 2)   # (batch_size, d_model, 100)
        
        x = self.convs(x)       # (batch_size, d_hidden*4, 1)
        x = x.squeeze(-1)       # (batch_size, d_hidden*4)
        
        x = self.dropout(x)
        x = self.fc(x)          # (batch_size, num_classes)
        return x
