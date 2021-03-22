# -*- coding: utf-8 -*-
# @Time    : 2019/8/24 12:06
# @Author  : zhoujun
import argparse

import torch
from torchvision import transforms
import os
import cv2
import time

from models import get_model

from post_processing import decode


def decode_clip(preds, scale=1, threshold=0.7311, min_area=5):
    import pyclipper
    import numpy as np
    preds[:2, :, :] = torch.sigmoid(preds[:2, :, :])
    preds = preds.detach().cpu().numpy()
    text = preds[0] > threshold  # text
    kernel = (preds[1] > threshold) * text  # kernel

    label_num, label = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=4)
    bbox_list = []
    for label_idx in range(1, label_num):
        points = np.array(np.where(label == label_idx)).transpose((1, 0))[:, ::-1]
        if points.shape[0] < min_area:
            continue
        rect = cv2.minAreaRect(points)
        poly = cv2.boxPoints(rect).astype(int)

        d_i = cv2.contourArea(poly) * 1.5 / cv2.arcLength(poly, True)
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrinked_poly = np.array(pco.Execute(d_i))
        if shrinked_poly.size == 0:
            continue
        rect = cv2.minAreaRect(shrinked_poly)
        shrinked_poly = cv2.boxPoints(rect).astype(int)
        if cv2.contourArea(shrinked_poly) < 800 / (scale * scale):
            continue

        bbox_list.append([shrinked_poly[1], shrinked_poly[2], shrinked_poly[3], shrinked_poly[0]])
    return label, np.array(bbox_list)


class Pytorch_model:
    def __init__(self, model_path, gpu_id=None):
        '''
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param gpu_id: 在哪一块gpu上运行
        '''
        self.gpu_id = gpu_id

        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")
        print('device:', self.device)
        checkpoint = torch.load(model_path, map_location=self.device)

        config = checkpoint['config']
        config['arch']['args']['pretrained'] = False
        self.net = get_model(config)

        self.img_channel = config['data_loader']['args']['dataset']['img_channel']
        self.net.load_state_dict(checkpoint['state_dict'])
        self.net.to(self.device)
        self.net.eval()

    def predict(self, img: str, short_size: int = 736):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img: 图像地址
        :param is_numpy:
        :return:
        '''
        assert os.path.exists(img), 'file is not exists'
        img = cv2.imread(img)
        oriImg = img
        if self.img_channel == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = short_size / min(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = transforms.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.to(self.device)
        with torch.no_grad():
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            start = time.time()
            preds = self.net(tensor)[0]
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            preds, boxes_list = decode(preds)
            scale = (preds.shape[1] / w, preds.shape[0] / h)
            if len(boxes_list):
                boxes_list = boxes_list / scale
            t = time.time() - start
        return preds, boxes_list, t, oriImg


def drawImg(imgPath, savePath=None):
    """
    绘制预测图片并保存(默认不存)
    :param imgPath: 图片路径
    :param savePath: 保存路经
    """
    mask, boxes_list, t = model.predict(imgPath)
    show_img(mask)
    if savePath:
        maskPath = savePath + imgPath[imgPath.rfind("/") + 1:-4] + "_mask.jpg"
        plt.savefig(maskPath, dpi=300)
    img = draw_bbox(cv2.imread(imgPath)[:, :, ::-1], boxes_list)
    show_img(img, color=True)
    if savePath:
        predPath = savePath + imgPath[imgPath.rfind("/") + 1:-4] + "_pred.jpg"
        plt.savefig(predPath, dpi=300)
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils.util import show_img, draw_bbox

    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--model', help='model file path', type=str,
                        default="/media/oldzhang/Data&Model&Course/model/251whole600.pth")
    parser.add_argument('--imgFiles', help="img folder path", type=str, default=None)
    parser.add_argument('--img', help='img file path', type=str,
                        default="/media/oldzhang/Data&Model&Course/data/ContainerNumber/reSizeImg/AMFU8782086.jpg")
    parser.add_argument("--save", help="save path", type=str, default=None)
    args = parser.parse_args()
    model = Pytorch_model(args.model, gpu_id=0)
    if args.imgFiles:
        imgs = sorted(os.listdir(args.imgFiles))
        for img in imgs:
            drawImg(args.imgFiles + img, args.save)
    if args.img:
        drawImg(args.img, args.save)
