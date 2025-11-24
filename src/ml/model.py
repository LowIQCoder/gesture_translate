import torch
import torch.nn as nn

class GestureModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 36,
        d_model: int = 84,
        d_hidden: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Treat the 84 landmarks as a 1D sequence
        self.convs = nn.Sequential(
            # Input: (batch, 1, 84)
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 84  42
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 42  21
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 21 1
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, 84)
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, 84)
        x = self.convs(x)   # (batch_size, 128, 1)
        x = x.view(x.size(0), -1)  # (batch_size, 128)
        x = self.classifier(x)  # (batch_size, num_classes)
        return x

if __name__ == "__main__":
    model = GestureModel()

    # Single sample: (batch_size=1, features=84)
    example_input = torch.rand((1, 84))
    logits = model(example_input)

    print(f"Input shape: {example_input.shape}")
    print(f"Logits shape: {logits.shape}")