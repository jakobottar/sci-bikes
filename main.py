import torch
from torch import nn
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models

import matplotlib.pyplot as plt
import numpy as np

import helper
import trainer
import logging
import progressbar

import argparse

class MyVGG(nn.Module):
    def __init__(self):
        super(MyVGG, self).__init__()
        self.flatten = nn.Flatten()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=1000, bias=True)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


progressbar.streams.wrap_stderr()
logging.basicConfig(filename="ref-vgg.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

logger=logging.getLogger()
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=int, default=0, help="GPU device number")
parser.add_argument('--epochs', type=int, default=5, help="Number of training/testing epochs")

FLAGS, unparsed = parser.parse_known_args()

GPU_NUM = FLAGS.gpu

if not torch.cuda.is_available():
    print("Looks like there's no CUDA devices available, sorry!")
    exit

device = "cuda:" + str(GPU_NUM)
logger.info(f"Using {device} device")

transform = {
    'train': transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ColorJitter(brightness=0.5),
                                transforms.ToTensor(), 
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'test': transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }
        
train_dataset = helper.BikeDataset('dvc_train.csv', img_dir='dvc/train', transform=transform["train"])
test_dataset = helper.BikeDataset('dvc_test.csv', img_dir='dvc/test', transform=transform["test"])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

model = models.vgg16(pretrained=False)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
# model = MyVGG()

logger.info("finished loading data and model")

model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

trainer = trainer.Trainer(model, loss_fn, optimizer, device)

trainer.SetEpochs(FLAGS.epochs)
trainer.Run(train_dataloader, test_dataloader, verbose=False)

# trainer.RunTrainer(train_dataloader, FLAGS.epochs)
# trainer.RunTester(test_dataloader)

# torch.save(model.state_dict(), "catsanddogs.pth")
# print("Saved PyTorch Model State to catsanddogs.pth")