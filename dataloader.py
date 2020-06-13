import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
import os
from PIL import Image
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


class ChestXrayDataset(Dataset):   
    def __init__(self, transform=transforms.ToTensor(), color='L'):
        self.transform = transform
        self.color = color
        self.image_dir = "/datasets/ChestXray-NIHCC/images/"
        self.image_info = pd.read_csv("/datasets/ChestXray-NIHCC/Data_Entry_2017.csv")
        self.image_filenames = self.image_info["Image Index"]
        self.labels = self.image_info["Finding Labels"]
        self.classes = {0: "Atelectasis", 1: "Cardiomegaly", 2: "Effusion", 
                3: "Infiltration", 4: "Mass", 5: "Nodule", 6: "Pneumonia", 
                7: "Pneumothorax", 8: "Consolidation", 9: "Edema", 
                10: "Emphysema", 11: "Fibrosis", 
                12: "Pleural_Thickening", 13: "Hernia"}

        
    def __len__(self):
        return len(self.image_filenames)


    def __getitem__(self, ind):
        image_path = os.path.join(self.image_dir, self.image_filenames.ix[ind])
        image = Image.open(image_path).convert(mode=str(self.color))
        if self.transform is not None:
            image = self.transform(image)
        if type(image) is not torch.Tensor:
            image = transform.ToTensor(image)
        label = self.convert_label(self.labels[ind], self.classes)
        
        return (image, label)

    

    def convert_label(self, label, classes):
        binary_label = torch.zeros(len(classes))
        for key, value in classes.items():
            if value in label:
                binary_label[key] = 1.0
        return binary_label
    
    

def create_split_loaders(batch_size, seed, transform=transforms.ToTensor(),
                         p_val=0.1, p_test=0.2, shuffle=True, 
                         show_sample=False, extras={}):

    dataset = ChestXrayDataset(transform)
    dataset_size = (int)(len(dataset) / 10)
    print("Dataset size:", dataset_size)
    all_indices = list(range(dataset_size))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(all_indices)
 
    val_split = int(np.floor(p_val * dataset_size))
    train_ind, val_ind = all_indices[val_split :], all_indices[: val_split]
    test_split = int(np.floor(p_test * len(train_ind)))
    train_ind, test_ind = train_ind[test_split :], train_ind[: test_split]
    
    sample_train = SubsetRandomSampler(train_ind)
    sample_test = SubsetRandomSampler(test_ind)
    sample_val = SubsetRandomSampler(val_ind)

    num_workers = 0
    pin_memory = False
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]
       
    train_loader = DataLoader(dataset, batch_size=batch_size, 
                              sampler=sample_train, num_workers=num_workers, 
                              pin_memory=pin_memory)

    test_loader = DataLoader(dataset, batch_size=batch_size, 
                             sampler=sample_test, num_workers=num_workers, 
                              pin_memory=pin_memory)

    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sample_val, num_workers=num_workers, 
                              pin_memory=pin_memory)

    print(len(train_loader), len(test_loader), len(val_loader), len(batch_size))
    return (train_loader, val_loader, test_loader)