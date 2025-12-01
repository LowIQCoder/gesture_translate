import torch
import torch.nn as nn

model_example_input = torch.rand((1, 84), dtype=torch.float32)

class GestureModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 36,
        d_model: int = 84,
        d_hidden: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.d_model = d_model

        # Treat the 84 landmarks as a 1D sequence
        self.convs = nn.Sequential(
            # Input: (batch, 1, 84)
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 84 42
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 42 21

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 21 10
            
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool1d(1)  # 10 1
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 + self.d_model, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(d_hidden, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(d_hidden, num_classes)
        )
    
    def forward(self, x_skip):
        # x shape: (batch_size, 84)
        x = x_skip.unsqueeze(1)  # Add channel dimension: (batch_size, 1, 84)
        x = self.convs(x)   # (batch_size, 128, 1)
        x = x.view(x.size(0), -1)  # (batch_size, 128)

        x = self.classifier(torch.cat([x, x_skip], dim=1))  # Concatenate along feature dimension
        return x

if __name__ == "__main__":
    model = GestureModel()

    logits = model(model_example_input)

    print(f"Input shape: {model_example_input.shape}")
    print(f"Logits shape: {logits.shape}")
