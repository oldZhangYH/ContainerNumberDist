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
    parmaeter.requires_grad = False
model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 4))

BatchSize = 32
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Epochs = 5
basePath = "I:/ContainerNumber/"

trainData = MyDataset(basePath)
trainLoader = DataLoader(trainData, BatchSize, shuffle=True)
optimizer = optim.SGD(model.parameters(), 1, 0.8)


def train(model, optimizer, trainLoader, epoch, Device):
    model.eval()
    model.to(Device)
    for (data, label) in trainLoader:
        data, label = data.to(Device), label.to(Device)
        optimizer.zero_grad()
        data = data.permute(0, 3, 1, 2)
        y = model(data)
        cost = (1 - iou(y, label)).sum()
        cost.backward()
        optimizer.step()

        print(cost)


def test():
    pass


for epoch in range(Epochs):
    train(model, optimizer, trainLoader, epoch, Device)
print("debug")
