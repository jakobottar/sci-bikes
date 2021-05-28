# % matplotlib inline

import torch
import os
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms, models
import torchdata as td

import matplotlib.pyplot as plt
import numpy as np

GPU_NUM = 0

if not torch.cuda.is_available():
    print("Looks like there's no CUDA devices available, sorry!")
    exit

device = "cuda:" + str(GPU_NUM)
print("Using {} device".format(device))

transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

data = td.datasets.WrapDataset(datasets.ImageFolder('dvc', transform=transform))

total_count = len(data)
train_count = int(0.8 * total_count)
test_count = total_count - train_count

train_data, test_data = torch.utils.data.random_split(data, (train_count, test_count)) # this can be extended to include a validation set too

train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)

# images, labels = next(iter(dataloader))
# im_ = images[0].squeeze()[0]
# plt.imshow(im_, cmap='gray')
# plt.show()

model = models.vgg16(pretrained=True).to(device)

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

        if batch % 100 == 0:
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

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print("Done!")

torch.save(model.state_dict(), "catsanddogs.pth")
print("Saved PyTorch Model State to catsanddogs.pth")