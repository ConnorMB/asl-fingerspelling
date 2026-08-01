import torch.nn as nn

NUM_LANDMARK_FEATURES = 63
NUM_CLASSES = 26


class ASLClassifier(nn.Module):
    def __init__(self, input_size: int = NUM_LANDMARK_FEATURES, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)
