import torch
from torch import nn
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models

import matplotlib.pyplot as plt
import numpy as np

import helper

GPU_NUM = 0

if not torch.cuda.is_available():
    print("Looks like there's no CUDA devices available, sorry!")
    exit

device = "cuda:" + str(GPU_NUM)
print("Using {} device".format(device))

transform = {
    'train': transforms.Compose([transforms.Resize(224),
                                transforms.ColorJitter(brightness=0.5),
                                transforms.ToTensor(), 
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'test': transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }
        
train_dataset = helper.BikeDataset('bikes.csv', img_dir='data/bikes_train', transform=transform["train"])
test_dataset = helper.BikeDataset('bikes.csv', img_dir='data/bikes_test', transform=transform["test"])

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# model = models.vgg16(pretrained=False)
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)

model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(batch)

        if batch % 16 == 0:
        # if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 8
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print("Done!")

torch.save(model.state_dict(), "bikes.pth")
print("Saved PyTorch Model State to bikes.pth")