import os
import torch
import numpy as np
import time
import cv2
import random
import torchvision
from models.resnet_yolo import resnet50
from models.yolo import YoloV1
import torchvision.transforms as transforms


class DetectModel(object):
    def __init__(self, ckpt_path, net_type='res_yolo'):
        # 保存信息
        self.ckpt_info = torch.load(ckpt_path)
        self.img_size = self.ckpt_info['img_size']

        # 模型
        self.in_channels = self.ckpt_info['in_channels']
        self.num_classes = self.ckpt_info['num_classes']
        self.S = self.ckpt_info['grid_num']
        self.B = self.ckpt_info['box_num']

        if net_type == 'res_yolo':
            self.net = resnet50()
            self.net.load_state_dict(self.ckpt_info['state_dict'])
        else:
            self.net = YoloV1(self.img_size, self.in_channels, self.num_classes, self.S, self.B)

        # 类别
        self.names = self.ckpt_info['names']

    def detect_dir(self, img_txt, draw_img, save_dir, save_txt):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        img_txt = open(img_txt, 'r')

        for line in img_txt.readlines():
            img_path = line.strip().split(' ')[0]
            img_data = cv2.imread(img_path)
            w, h, _ = img_data.shape
            if draw_img:
                preds, img_res = self.detect_image(img_data, draw_img)
            else:
                preds = self.detect_image(img_data, draw_img)

            if save_txt:
                save_txt_ = open(os.path.join(save_dir, os.path.basename(img_path).split('.')[0] + '.txt'), 'w')
                for pred in preds:
                    pred_str = []
                    pred_str.append(str(int(pred[5])))
                    pred_str.append(str(round(pred[0] / w, 4)))
                    pred_str.append(str(round(pred[1] / h, 4)))
                    pred_str.append(str(round(pred[2] / w, 4)))
                    pred_str.append(str(round(pred[3] / h, 4)))
                    pred_str.append(str(pred[4]))
                    save_txt_.writelines(' '.join(pred_str) + '\n')     # C, x,y,x,y,prob

    def detect_image(self, image, draw_img):
        image_copy = image.copy()
        image_copy = image_copy
        w, h, _ = image.shape
        img_size = (w, h)

        img_data = self.preprocess(image, [transforms.ToTensor()])
        img_data = img_data.unsqueeze(0)
        with torch.no_grad():
            self.net.eval()
            output = self.net(img_data)
        prediction = self.postprocess(img_size, output, conf_thres=0.1, iou_thres=0.5)

        res_boxs = []
        res_labels = []
        for i, det in enumerate(prediction):  # detections per image
            # det x,y,w,h,conf,cls_id, probs
            if len(det):
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    res_boxs.append([int(xyxy[0]),
                                     int(xyxy[1]),
                                     int(xyxy[2]),
                                     int(xyxy[3])])

                    res_labels.append(self.names[int(cls)])
                    if draw_img:  # Add bbox to image
                        label = self.names[int(cls)] + ' ' + str(round(float(conf), 2))
                        DetectModel.plot_one_box(xyxy, image_copy, label=label, color=(0, 0, 255), line_thickness=1)

                    label = self.names[int(cls)] + ' ' + str(round(float(conf), 2))
                    DetectModel.plot_one_box(xyxy, image_copy, label=label, color=(0, 0, 255), line_thickness=1)
                    cv2.imwrite('res.jpg', image_copy)
        prediction = prediction[0].detach().cpu().numpy()

        if draw_img:
            return prediction, image_copy
        else:
            return prediction

    def preprocess(self, img, transform):
        mean = (123, 117, 104)  # RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # because pytorch pretrained model use RGB
        img = self.subMean(img, mean)  # 减去均值
        img = cv2.resize(img, (self.img_size, self.img_size))
        for t in transform:
            img = t(img)
        return img

    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def postprocess(self, img_size, output, conf_thres, iou_thres):
        # output: [BS, S, S, 5*B+C]
        prediction = self.decode(output, img_size)
        # prediction: [BS, S*S*B, 5+C] ==> x1,y1,x2,y2,C,prob1,prob2,...,probC
        prediction = DetectModel.non_max_suppression(prediction, conf_thres=conf_thres, iou_thres=iou_thres)
        return prediction

    def decode(self, output, size):
        # output 7 * 7 * 30
        output = output.squeeze(0)
        cell_size = 1.0 / float(self.S)
        yv, xv = torch.meshgrid([torch.arange(self.S), torch.arange(self.S)], indexing="ij")
        grids = torch.stack((xv, yv), 2)
        grids_x0y0 = grids * cell_size
        for k in range(self.B):
            s = 5 * k
            xy_normalized = output[..., s:s+2] * cell_size
            xy = xy_normalized + grids_x0y0
            output[..., s:s+2] = xy
            output[..., s] *= size[0]
            output[..., s+1] *= size[1]
            output[..., s+2] *= size[0]
            output[..., s+3] *= size[1]

        # bs * S * S * 30 ==> N * (5 + C)
        prediction = torch.zeros(self.S * self.S * self.B, 5 + self.num_classes)
        t = 0
        for i in range(self.S):
            for j in range(self.S):
                for k in range(self.B):
                    s = 5 * k
                    box_info = output[i, j, s:s+4]
                    conf = output[i, j, s+4]
                    cls_probs = output[i, j, 2*5:]
                    conf = torch.tensor(np.array([conf]))
                    prediction[t] = torch.cat([box_info, conf, cls_probs], 0)
                    t += 1

        prediction[:, :4] = self.xywh2xyxy(prediction[:, :4]).clamp(0, self.img_size)
        prediction = prediction.unsqueeze(0)

        return prediction

    @staticmethod
    def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
        """Performs Non-Maximum Suppression (NMS) on inference results

        Returns:
             detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
        """
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_det = 300  # maximum number of detections per image
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 5), device=x.device)
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            print(x[:, 4:5].shape)
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf #############################################

            # get (x1, y1, x2, y2)
            box = x[:, :4]

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = DetectModel.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit}s exceeded')
                break  # time limit exceeded
        return output

    @staticmethod
    def xywh2xyxy(boxes):
        """
        :param boxes: (Tensor[N, 4])
        :return:
        """
        boxes_ = torch.zeros(boxes.shape) if isinstance(boxes, torch.Tensor) else np.zeros(boxes.shape)
        boxes_[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return boxes_.clamp(min=0)

    @staticmethod
    def box_iou(box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """
        # print(box1.size(), box2.size())
        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

    @staticmethod
    def plot_one_box(x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


if __name__ == '__main__':

    ckpt_path = 'checkpoints/yolov1.pt'
    detector = DetectModel(ckpt_path)

    img_data = cv2.imread(r'resources/scratches_42.jpg')
    prediction, res_img = detector.detect_image(img_data, True)

    cv2.imwrite('resources/result.jpg', res_img)







