import torch
import torch.nn.functional as F
import torchvision
import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import tqdm
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self, input_dim=2048) -> None:
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 256)
        )
    
    def forward(self, x):
        return self.net(x)
    

class OnlineModel(nn.Module):
    def __init__(self) -> None:
        super(OnlineModel, self).__init__()
        self.encoder = torchvision.models.resnet50()
        self.encoder.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.fc = nn.Identity()
        self.encoder.maxpool = torch.nn.Identity()

        self.represent = MLP()

        # self.predictor = MLP(input_dim=256)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.represent(x)
        # x = self.predictor(x)
        return x