import os
from json import load
import cv2
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, datapath):
        self.datapath = datapath
        img = os.listdir(self.datapath + "reSizeImg/")  # 目录里的所有文件
        df = []
        for i in img:
            df.append(datapath + "reSizeImg/" + i)
        self.df = df

        self.l, self.s = getLabelFromJson(datapath + "label/reLabel/")
        self.l = torch.Tensor(self.l)
        self.s = torch.Tensor(self.s)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = cv2.imread(self.df[idx])
        # img = cv2.resize(img, (450, 800))
        img = torch.Tensor(img)
        l = self.l[idx]
        s = self.s[idx]
        return img, l, s


def iou(y, label):




def reSize(inPath, outPath):
    imgList = os.listdir(inPath)
    for i in imgList:
        img = cv2.imread(inPath + i)
        img = cv2.resize(img, (450, 800))
        cv2.imwrite(outPath + i, img)
    print("debug")


def getLabelFromJson(jsonPath):
    """
    :param jsonPath:json文件夹路径
    :return:[l,s]
    """
    jsonData = os.listdir(jsonPath)
    l, s = [], []
    for i in jsonData:
        with open(jsonPath + i, "r") as f:
            json = load(f)
            templ, temps = [], []
            for j in range(0, len(json["shapes"][0]["points"])):
                if json["shapes"][0]["label"] == 'l':
                    templ.extend(json["shapes"][0]["points"][j])
                else:
                    templ.extend(json["shapes"][1]["points"][j])
            for j in range(0, len(json["shapes"][1]["points"])):
                if json["shapes"][0]["label"] == 's':
                    temps.extend(json["shapes"][1]["points"][j])
                else:
                    temps.extend(json["shapes"][0]["points"][j])
        l.append(templ)
        s.append(temps)
    return l, s

# getLabelFromJson("I:/ContainerNumber/label/reLabel/")
# reSize("I:/ContainerNumber/img/", "I:/ContainerNumber/reSizeImg/")
