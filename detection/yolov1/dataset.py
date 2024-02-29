'''
txt描述文件 image_name.jpg x y w h c x y w h c 这样就是说一张图片中有两个目标
'''
import os
import sys
import os.path

import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import cv2
import matplotlib.pyplot as plt


def get_train_transforms():
    return transforms.ToTensor()


def get_valid_transforms():
    return transforms.ToTensor()


class YOLODataset(data.Dataset):
    image_size = 448

    def __init__(self, data_txt_path, S, B, C, transforms=None, train=True):
        self.data_list = open(data_txt_path, 'r').readlines()
        self.transforms = transforms
        self.train = train
        self.S = S
        self.B = B
        self.C = C

        self.boxes = []
        self.labels = []
        self.mean = (123, 117, 104)  # RGB

    def __getitem__(self, idx):
        line_data = self.data_list[idx].strip()
        img_path, label_path = line_data.split(' ')[0], line_data.split(' ')[1]

        '''图像'''
        img_data = cv2.imread(img_path)

        '''标签'''
        label_txt = open(label_path, 'r')
        box_list, label_list = [], []
        for line in label_txt.readlines():
            line = line.strip().split()
            line_list = [float(i) for i in line]
            box_list.append(line_list[1:])
            label_list.append(line_list[0])
        boxes = torch.tensor(np.array(box_list))
        labels = torch.tensor(np.array(label_list))

        # '''数据增强'''
        h, w, _ = img_data.shape
        boxes = self.xywh2xyxy_denorm(boxes, (w, h))
        if self.train and len(boxes):
            # x,y,w,h转化x1,y1,x2,y2
            # print('888')
            img_data, boxes = self.random_flip(img_data, boxes)
            img_data, boxes = self.randomScale(img_data, boxes)
            img_data = self.randomBlur(img_data)
            img_data = self.RandomBrightness(img_data)
            img_data = self.RandomHue(img_data)
            img_data = self.RandomSaturation(img_data)
            img_data, boxes, labels = self.randomShift(img_data, boxes, labels)
            img_data, boxes, labels = self.randomCrop(img_data, boxes, labels)
        else:
            pass
        h, w, _ = img_data.shape
        # #debug
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)
        # xyxy2xywh
        boxes = self.xyxy2xywh_norm(boxes)
        img = self.BGR2RGB(img_data)  # because pytorch pretrained model use RGB
        img_data = self.subMean(img, self.mean)  # 减去均值
        img_data = cv2.resize(img_data, (self.image_size, self.image_size))

        '''数据处理'''
        if self.transforms:
            img_data = self.transforms(img_data)

        '''标签编码'''
        target, grid_mask_obj = self.encode(boxes, labels)  # x,y,w,h norm
        return img_data, target, grid_mask_obj

    def __len__(self):
        return len(self.data_list)

    def encode(self, boxes, labels):
        # 将label编码为s * s * (2 * B + C)
        target = torch.zeros(self.S, self.S, 5 * self.B + self.C)
        grid_mask_obj = torch.zeros(self.S, self.S)
        cell_size = 1.0 / float(self.S)
        for b in range(boxes.shape[0]):
            xy, wh, label = boxes[b][:2], boxes[b][2:4], int(labels[b])
            mn = (xy / cell_size).ceil() - 1.0  # 从0开始计数
            m, n = int(mn[0]), int(mn[1])
            grid_mask_obj[n][m] = 1.0
            # 网格左上角
            x0y0 = mn * cell_size
            # 相对于网格左上角的坐标（归一化后的）
            xy_normalized = (xy - x0y0) / cell_size
            for k in range(self.B):
                s = 5 * k
                target[n, m, s:s + 2] = xy_normalized
                target[n, m, s + 2:s + 4] = wh
                target[n, m, s + 4] = 1.0
            target[n, m, self.B * 5 + label] = 1.0
        return target, grid_mask_obj

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def RandomBrightness(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomHue(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self, bgr):
        if random.random() < 0.5:
            bgr = cv2.blur(bgr, (5, 5))
        return bgr

    def randomShift(self, bgr, boxes, labels):
        # 平移变换
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        if random.random() < 0.5:
            height, width, c = bgr.shape
            after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
            shift_x = random.uniform(-width * 0.2, width * 0.2)
            shift_y = random.uniform(-height * 0.2, height * 0.2)
            # 原图像的平移
            if shift_x >= 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, int(shift_x):, :] = bgr[:height - int(shift_y), :width - int(shift_x),
                                                                     :]
            elif shift_x >= 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, :width - int(shift_x),
                                                                              :]
            elif shift_x < 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, :width + int(shift_x), :] = bgr[:height - int(shift_y), -int(shift_x):,
                                                                             :]
            elif shift_x < 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), :width + int(shift_x), :] = bgr[-int(shift_y):,
                                                                                      -int(shift_x):, :]

            shift_xy = torch.FloatTensor([[int(shift_x), int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
            mask = (mask1 & mask2).view(-1, 1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(boxes_in)
            boxes_in = boxes_in + box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image, boxes_in, labels_in
        return bgr, boxes, labels

    def randomScale(self, bgr, boxes):
        # 固定住高度，以0.8-1.2伸缩宽度，做图像形变
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            height, width, c = bgr.shape
            bgr = cv2.resize(bgr, (int(width * scale), height))
            scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr, boxes
        return bgr, boxes

    def randomCrop(self, bgr, boxes, labels):
        if random.random() < 0.5:
            center = (boxes[:, 2:] + boxes[:, :2]) / 2
            height, width, c = bgr.shape
            h = random.uniform(0.6 * height, height)
            w = random.uniform(0.6 * width, width)
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)
            x, y, h, w = int(x), int(y), int(h), int(w)

            center = center - torch.FloatTensor([[x, y]]).expand_as(center)
            mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
            mask = (mask1 & mask2).view(-1, 1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if (len(boxes_in) == 0):
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
            boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
            boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
            boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y + h, x:x + w, :]
            return img_croped, boxes_in, labels_in
        return bgr, boxes, labels

    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape

            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return im, boxes

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta, delta)
            im = im.clip(min=0, max=255).astype(np.uint8)
        return im

    def xywh2xyxy_denorm(self, boxes, size):
        boxes_ = torch.zeros(boxes.shape) if isinstance(boxes, torch.Tensor) else np.zeros(boxes.shape)
        boxes_[:, 0] = (boxes[:, 0] * size[0] - boxes[:, 2] * size[0] // 2).floor()
        boxes_[:, 1] = (boxes[:, 1] * size[1] - boxes[:, 3] * size[1] // 2).floor()
        boxes_[:, 2] = (boxes[:, 0] * size[0] + boxes[:, 2] * size[0] // 2).floor()
        boxes_[:, 3] = (boxes[:, 1] * size[1] + boxes[:, 3] * size[1] // 2).floor()
        return boxes_

    def xyxy2xywh_norm(self, boxes):
        boxes_ = torch.zeros(boxes.shape) if isinstance(boxes, torch.Tensor) else np.zeros(boxes.shape)
        boxes_[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
        boxes_[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
        boxes_[:, 2] = (boxes[:, 2] - boxes[:, 0])
        boxes_[:, 3] = (boxes[:, 3] - boxes[:, 1])
        return boxes_


if __name__ == '__main__':
    from dataset import *

    data_txt_path = r'C:\Users\Administrator\Desktop\my_paper\code\yolov1\data\NEU-DET\train.txt'
    yolo_dataset = YOLODataset(data_txt_path, 7, 2, 6, get_train_transforms(), True)

    for i in range(1000):
        img_data, target, _ = yolo_dataset.__getitem__(i)
        print(img_data.shape)




