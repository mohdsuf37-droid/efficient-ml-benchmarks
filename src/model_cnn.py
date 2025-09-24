import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyMNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)   # 28x28 -> 26x26
        self.conv2 = nn.Conv2d(16, 32, 3, 1)  # 26x26 -> 24x24
        # maxpool(2) -> 12x12
        self.fc1 = nn.Linear(32 * 12 * 12, 64)  # 32*12*12 = 4608
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))            # [B,16,26,26]
        x = F.relu(self.conv2(x))            # [B,32,24,24]
        x = F.max_pool2d(x, 2)               # [B,32,12,12]
        x = torch.flatten(x, 1)              # [B, 4608]
        x = F.relu(self.fc1(x))              # [B, 64]
        x = self.fc2(x)                      # [B, 10]
        return x
