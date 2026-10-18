import torch.nn as nn

class WineNet(nn.Module):
    def __init__(self):
        super(WineNet, self).__init__()
        self.fc1 = nn.Linear(13, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
