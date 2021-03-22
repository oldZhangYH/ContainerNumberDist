# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 10:57
# @Author  : zhoujun
import os
import sys
import pathlib

# 将 torchocr路径加到python陆经里
__dir__ = pathlib.Path(os.path.abspath(__file__))

from time import time

sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from torchocr.networks import build_model
from torchocr.datasets.RecDataSet import RecDataProcess
from torchocr.utils import CTCLabelConverter


class RecInfer:
    def __init__(self, model_path, alphabets):
        ckpt = torch.load(model_path, map_location='cpu')
        cfg = ckpt['cfg']
        cfg["dataset"]['alphabet'] = alphabets
        self.model = build_model(cfg['model'])
        state_dict = {}
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        self.model.load_state_dict(state_dict)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        self.process = RecDataProcess(cfg['dataset']['train']['dataset'])
        self.converter = CTCLabelConverter(cfg['dataset']['alphabet'])

    def predict(self, img):
        # 预处理根据训练来
        img = self.process.resize_with_specific_height(img)
        # img = self.process.width_pad_img(img, 120)
        img = self.process.normalize_img(img)
        tensor = torch.from_numpy(img.transpose([2, 0, 1])).float()
        tensor = tensor.unsqueeze(dim=0)
        tensor = tensor.to(self.device)
        out = self.model(tensor)
        txt = self.converter.decode(out.softmax(dim=2).detach().cpu().numpy())
        return txt


def getCrop(box):
    h, w = [], []
    box = box.T
    w.append(min(box[0]))
    w.append(max(box[0]))
    h.append(min(box[1]))
    h.append(max(box[1]))
    return h, w


def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='PytorchOCR infer')
    parser.add_argument('--recModel', type=str, help='rec model path', default="/media/oldzhang/Data&Model&Course"
                                                                               "/model/CE_crnn_res34.pth")
    parser.add_argument('--decModel', help='model file path', type=str,
                        default="/media/oldzhang/Data&Model&Course/model/251whole600.pth")
    parser.add_argument('--img_path', type=str, help='img path for predict',
                        default="/media/oldzhang/Data&Model&Course/data/ContainerNumber/reSizeImg/AMFU8782086.jpg")
    parser.add_argument("--alphabets", type=str, help="alphabets path", default="torchocr/datasets/alphabets"
                                                                                "/ppocr_keys_v1.txt")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import cv2

    star = time()
    from decePredict import Pytorch_model

    args = init_args()
    decModel = Pytorch_model(args.decModel, gpu_id=0)
    _, box, decT, img = decModel.predict(args.img_path)

    imgs = []
    box = box.astype(np.uint32)
    for i in box:
        h, w = getCrop(i)
        imgs.append(img[h[0]:h[1], w[0]:w[1]])
    model = RecInfer(args.recModel, args.alphabets)
    for img in imgs:
        out = model.predict(img)
        print(out[0][0])
    print(time() - star)
