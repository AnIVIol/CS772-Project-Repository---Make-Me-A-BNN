import torch
import torch.nn as nn
import torch.nn.functional as F
import bnl
from bnl import BNL

class Net(nn.Module):
    def __init__(self, norm_type='batchnorm'):
        super(Net, self).__init__()

        if norm_type == 'batchnorm':
            norm_layer = lambda num_features: nn.BatchNorm2d(num_features)
            norm_fc_layer = lambda num_features: nn.BatchNorm1d(num_features)
        elif norm_type == 'bnl':
            norm_layer = lambda num_features: bnl.BNL(num_features)
            norm_fc_layer = lambda num_features: bnl.BNL(num_features)
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")

        # Conv layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.norm1 = norm_layer(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.norm2 = norm_layer(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.norm3 = norm_layer(128)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # FC layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.norm4 = norm_fc_layer(256)

        self.fc2 = nn.Linear(256, 128)
        self.norm5 = norm_fc_layer(128)

        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        x = self.pool(F.relu(self.norm2(self.conv2(x))))
        x = self.pool(F.relu(self.norm3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.norm4(self.fc1(x)))
        x = F.relu(self.norm5(self.fc2(x)))
        x = self.fc3(x)
        return x
    
    
class ABNN_mixed(nn.Module):
    def __init__(self, norm_type='batchnorm'):
        super(ABNN_mixed, self).__init__()

        if norm_type == 'batchnorm':
            norm_layer = lambda num_features: nn.BatchNorm2d(num_features)
            norm_layer_bnl = lambda num_features: nn.BatchNorm2d(num_features)
            norm_fc_layer = lambda num_features: nn.BatchNorm1d(num_features)
        elif norm_type == 'bnl':
            norm_layer = lambda num_features: nn.BatchNorm2d(num_features)
            norm_layer_bnl = lambda num_features: BNL(num_features)
            norm_fc_layer = lambda num_features: BNL(num_features)
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")
            
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.norm1 = norm_layer(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.in2 = norm_layer_bnl(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.ln3 = norm_layer_bnl(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.ln4 = norm_fc_layer(256)
        self.fc2 = nn.Linear(256, 128)
        self.norm5 = norm_fc_layer(128)
        self.fc3 = nn.Linear(128, 10)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.in2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.ln3(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, 128 * 4 * 4)

        x = self.fc1(x)
        x = self.ln4(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.norm5(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
