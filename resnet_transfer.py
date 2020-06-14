import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim
import torchvision
from torchvision import transforms, utils
from dataloader import ChestXrayDataset, create_split_loaders
import matplotlib.pyplot as plt
import numpy as np
import os

class Resnet_18_Transfer(nn.Module):
    def __init__(self, num_classes, fine_tuning = False):
        super(Resnet_18_Transfer, self).__init__()
        res = torchvision.models.resnet18(pretrained = True)
        for param in res.parameters():
            param.requires_grad = fine_tuning

        self.conv1 = res.conv1
        self.bn1 = res.bn1
        self.relu = res.relu
        self.maxpool = res.maxpool

        self.layer1 = res.layer1
        self.layer2 = res.layer2
        self.layer3 = res.layer3
        self.layer4 = res.layer4
        self.avgpool = res.avgpool

        num_ftrs = res.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):

        f = self.conv1(x)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        f = self.avgpool(f)
        f = f.reshape(f.size(0), -1)
        y = self.fc(f)
        return y