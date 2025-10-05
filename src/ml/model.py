import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(84, 128),
            nn.ReLU(),
            nn.Linear(128, 2560),
            nn.ReLU(),
            nn.Linear(2560, 512),
            nn.ReLU(),
            nn.Linear(512, 43)
        )

    def forward(self, x):
        return self.classifier(x)
