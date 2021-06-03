import torch.nn as nn
import torch.nn.functional as F

__all__ = ['fcnet']


class FCNet(nn.Module):

    def __init__(self, num_classes=10):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(3072, 1176)
        self.fc2 = nn.Linear(1176, 400)
        self.fc3 = nn.Linear(16*5*5, 120)
        self.fc4 = nn.Linear(120, 84)
        self.fc5 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = self.fc5(out)
        return out


def fcnet(**kwargs):
    model = FCNet(**kwargs)
    return model
