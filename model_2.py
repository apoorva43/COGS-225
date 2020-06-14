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
import matplotlib
import numpy as np
import os

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv1_normed = nn.BatchNorm2d(64)
        torch_init.xavier_normal_(self.conv1.weight)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.conv2_normed = nn.BatchNorm2d(64)
        torch_init.xavier_normal_(self.conv2.weight)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.conv3_normed = nn.BatchNorm2d(64)
        torch_init.xavier_normal_(self.conv3.weight)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=64*64*64, out_features=256)
        self.fc1_normed = nn.BatchNorm1d(256)
        torch_init.xavier_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc2_normed = nn.BatchNorm1d(128)
        torch_init.xavier_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(in_features=128, out_features=14)
        torch_init.xavier_normal_(self.fc3.weight)

    def forward(self, batch):
        
        batch = func.relu(self.conv1_normed(self.conv1(batch)))
        batch = func.relu(self.conv2_normed(self.conv2(batch)))
        batch = func.relu(self.conv3_normed(self.conv3(batch)))
        batch = self.pool1(batch)
        batch = self.pool2(batch)
        batch = batch.view(-1, self.num_flat_features(batch))
        batch = func.relu(self.fc1_normed(self.fc1(batch)))
        batch = func.relu(self.fc2_normed(self.fc2(batch)))
        batch = self.fc3(batch)
        r = torch.sigmoid(batch)
        return r

    def num_flat_features(self, inputs):

        size = inputs.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features
