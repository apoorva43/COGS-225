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

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=8)
        self.conv1_normed = nn.BatchNorm2d(12)
        torch_init.xavier_normal_(self.conv1.weight)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=10, kernel_size=8)
        self.conv2_normed = nn.BatchNorm2d(10)
        torch_init.xavier_normal_(self.conv2.weight)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=8, kernel_size=6)
        self.conv3_normed = nn.BatchNorm2d(8)
        torch_init.xavier_normal_(self.conv3.weight)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        self.fc1 = nn.Linear(in_features=164*164*8, out_features=128)
        self.fc1_normed = nn.BatchNorm1d(128)
        torch_init.xavier_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(in_features=128, out_features=14)
        self.fc2_normed = nn.BatchNorm1d(14)
        torch_init.xavier_normal_(self.fc2.weight)

    def forward(self, batch):
        
        batch = func.relu(self.conv1_normed(self.conv1(batch)))
        batch = func.relu(self.conv2_normed(self.conv2(batch)))
        batch = func.relu(self.conv3_normed(self.conv3(batch)))
        batch = self.pool(batch)
        batch = batch.view(-1, self.num_flat_features(batch))
        batch = func.relu(self.fc1_normed(self.fc1(batch)))
        batch = self.fc2(batch)
        r = torch.sigmoid(batch)
        return r 

    def num_flat_features(self, inputs):

        size = inputs.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features
