import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

import progressbar
import logging

# def SplitLoad(rootFolder, trainSplit, transforms):
#     dataSets = {}

#     data = td.datasets.WrapDataset(datasets.ImageFolder(rootFolder, transform=transforms["test"]))

#     total_count = len(data)
#     train_count = int(trainSplit * total_count)
#     test_count = total_count - train_count

#     dataSets["train"], dataSets["test"] = torch.utils.data.random_split(data, (train_count, test_count)) # this can be extended to include a validation set too

#     return dataSets

# TODO: I should be able to make this grab files listed in a file vs everything in the folder to solve the test/train split problem

class BikeDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(img_dir +  "/" + annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path) # TODO: how does Image.open() compare to torchvision.io.read_image()?
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label