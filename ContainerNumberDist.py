import os

from torch import optim
from torchvision import models
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

from Utils import MyDataset, iou

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = models.resnet50(pretrained=True)
for parmaeter in model.parameters():
    parmaeter.requires_grad = True
model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 512),
                               torch.nn.Linear(512, 64),
                               torch.nn.Linear(64, 16),
                               torch.nn.ReLU()
                               )

BatchSize = 4
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Epochs = 5
basePath = "I:/ContainerNumber/"

trainData = MyDataset(basePath)
trainLoader = DataLoader(trainData, BatchSize, shuffle=True)
optimizer = optim.SGD(model.parameters(), 1, 0.8)


def train(model, optimizer, trainLoader, epoch, Device):
    model.eval()
    model.to(Device)
    for (data, l, s) in trainLoader:
        data, l, s = data.to(Device), l.to(Device), s.to(Device)
        optimizer.zero_grad()
        data = data.permute(0, 3, 1, 2)
        y = model(data)
        costL = (1 - iou(y, l)).sum() / BatchSize
        costS = (1 - iou(y, l)).sum() / BatchSize
        cost = costS + costL
        cost.backward()
        optimizer.step()
        print(cost)


def test():
    pass


for epoch in range(Epochs):
    train(model, optimizer, trainLoader, epoch, Device)
print("debug")
