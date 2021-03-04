import os
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
        # labels = open(self.datapath + "label/label.txt")
        coordinates = open(self.datapath + "label/coordinate.txt")
        res = []
        # temp=[]
        # court=0
        # for label in labels:
        #     temp.append(label[-5:-1])
        str = ""
        temp = []
        for coordinate in coordinates:
            coordinate = coordinate[12:-1]
            for i in range(len(coordinate)):
                if coordinate[i] != ',' and coordinate[i] != ' ':
                    str += (coordinate[i])
                else:
                    temp.append(int(str))
                    str = ""
            res.append(temp)
            temp = []
        res = torch.Tensor(res)
        self.label = res

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = cv2.imread(self.df[idx])
        # img = cv2.resize(img, (450, 800))
        img = torch.Tensor(img)
        label = self.label[idx]
        return img, label


def iou(box1, box2):
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    if torch.cuda.is_available():
        inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape).cuda()) * torch.max(
            inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape).cuda())
    else:
        inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape)) * torch.max(
            inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape))

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def reSize(inPath, outPath):
    imgList = os.listdir(inPath)
    for i in imgList:
        img = cv2.imread(inPath + i)
        img = cv2.resize(img, (450, 800))
        cv2.imwrite(outPath + i, img)
    print("debug")


# reSize("I:/ContainerNumber/img/", "I:/ContainerNumber/reSizeImg/")
