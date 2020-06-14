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

class VGG_Transfer:
    def __init__(self, n_class, finetuning = False):
        self.finetuning = finetuning
        self.n_class = n_class

    def __call__(self, model):
        if not self.finetuning:
            self.gradFreeze(model)
        self.fcRest(model)
        return model

    def fcRest(self, pretrained):
        num_ins = pretrained.fc.in_features
        pretrained.fc = nn.Linear(num_ins, self.n_class)

    def gradFreeze(self, pretrained):
        for param in pretrained.parameters():
            param.requires_grad = False