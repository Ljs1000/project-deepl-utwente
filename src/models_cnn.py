import torch
import torch.nn as nn
import torch.nn.functional as F

class PneumoniaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.Dropout2d(0.10),
            nn.BatchNorm2d(64), nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.Dropout2d(0.20),
            nn.BatchNorm2d(128), nn.MaxPool2d(2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.Dropout2d(0.20),
            nn.BatchNorm2d(256), nn.MaxPool2d(2)
        )

        # input is 150x150 -> after 5 pools: 150/32 ≈ 4 (integer floor depends on pooling)
        # We'll infer at runtime by doing a dummy forward once, but simplest is use Adaptive pooling.
        self.gap = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.20)
        )
        self.fc_out = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gap(x)
        x = self.fc1(x)
        logits = self.fc_out(x)  # BCEWithLogitsLoss expects raw logits
        return logits
