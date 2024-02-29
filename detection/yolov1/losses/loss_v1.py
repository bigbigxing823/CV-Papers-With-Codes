import torch
import numpy as np
import torch.nn as nn


class YOLOV1LossV1(nn.Module):
    def __init__(self, gamma_coord, gamma_noobj, S, B, img_size):
        super().__init__()
        self.gamma_coord = gamma_coord
        self.gamma_noobj = gamma_noobj
        self.S = S
        self.B = B
        self.img_size = img_size

    def forward(self, output, target, grid_mask_obj):
        regression_loss_obj = self.regression_loss_obj(output, target, grid_mask_obj)

        confidence_loss_obj = self.confidence_loss_obj(output, target, grid_mask_obj)

        confidence_loss_noobj = self.confidence_loss_noobj(output, target, grid_mask_obj)

        classify_loss_obj = self.classify_loss_obj(output, target, grid_mask_obj)
        # print('regression_loss_obj', regression_loss_obj)
        # print('confidence_loss_obj', confidence_loss_obj)
        # print('confidence_loss_noobj', confidence_loss_noobj)
        # print('classify_loss_obj', classify_loss_obj)
        return regression_loss_obj + confidence_loss_obj + confidence_loss_noobj + classify_loss_obj

    def regression_loss_obj(self, output, target, grid_mask_obj):
        bs = grid_mask_obj.size(0)
        obj_ij_mask, _ = self.get_obj_ij_mask(output, target, grid_mask_obj)
        xy_loss_total, wh_loss_total = 0, 0
        for i in range(bs):
            grid_mask_obj_i = grid_mask_obj[i]
            obj_indices = torch.nonzero(grid_mask_obj_i != 0)
            xy_loss, wh_loss = 0, 0
            for nm in obj_indices:
                n, m = nm[0], nm[1]
                box_id = torch.argmax(obj_ij_mask[i, n, m, :])
                output_xy = [output[i, n, m, 5*t:5*t+2] for t in range(self.B)][int(box_id)]  # [4, 9]
                target_xy = [target[i, n, m, 5*t:5*t+2] for t in range(self.B)][int(box_id)]  # [4, 9]
                output_wh = [output[i, n, m, 5*t+2:5*t+4] for t in range(self.B)][int(box_id)]  # [4, 9]
                target_wh = [target[i, n, m, 5*t+2:5*t+4] for t in range(self.B)][int(box_id)]  # [4, 9]
                xy_loss += torch.sum(torch.pow(output_xy - target_xy, 2))
                wh_loss += torch.sum(torch.pow(torch.sqrt(output_wh) - torch.sqrt(target_wh), 2))
            xy_loss_total += xy_loss
            wh_loss_total += wh_loss
        return self.gamma_coord * (xy_loss_total + wh_loss_total) / bs

    def confidence_loss_obj(self, output, target, grid_mask_obj):
        bs = grid_mask_obj.size(0)
        _, obj_ij_ious = self.get_obj_ij_mask(output, target, grid_mask_obj)
        c_loss_obj_total = 0
        for i in range(bs):
            grid_mask_obj_i = grid_mask_obj[i]
            obj_indices = torch.nonzero(grid_mask_obj_i != 0)
            c_loss_obj = 0
            for nm in obj_indices:
                n, m = nm[0], nm[1]
                box_id = torch.argmax(obj_ij_ious[i, n, m, :])
                output_c = output[i, n, m, [4 + 5*t for t in range(self.B)]][box_id]  # [4, 9]  ####################
                # target_c = target[i, n, m, [4 + 5*t for t in range(self.B)]][box_id]  # [4, 9]  ####################
                # print('output_c', output_c, 'target_c', target_c)
                # c_loss_obj += torch.pow(output_c - target_c, 2) * obj_ij_ious[i, n, m, box_id] ###################
                c_loss_obj += torch.pow(output_c - obj_ij_ious[i, n, m, box_id], 2)  ##################
            c_loss_obj_total += c_loss_obj
        return c_loss_obj_total / bs

    def confidence_loss_noobj(self, output, target, grid_mask_obj):
        bs = grid_mask_obj.size(0)
        c_loss_noobj_total = 0
        for i in range(bs):
            grid_mask_obj_i = grid_mask_obj[i]
            noobj_indices = torch.nonzero(grid_mask_obj_i == 0)
            c_loss_noobj = 0
            for nm in noobj_indices:
                n, m = nm[0], nm[1]
                output_c = output[i, n, m, [4+t*5 for t in range(self.B)]]  # [4, 9]
                target_c = target[i, n, m, [4+t*5 for t in range(self.B)]]  # [4, 9]
                c_loss_noobj += torch.sum(torch.pow(output_c - target_c, 2))
            c_loss_noobj_total += c_loss_noobj
        return self.gamma_noobj * c_loss_noobj_total / bs

    def classify_loss_obj(self, output, target, grid_mask_obj):
        bs = grid_mask_obj.size(0)
        probs_loss = 0
        for i in range(bs):
            grid_mask_obj_i = grid_mask_obj[i]
            obj_indices = torch.nonzero(grid_mask_obj_i != 0)
            for nm in obj_indices:
                n, m = nm[0], nm[1]
                output_probs = output[i, n, m, self.B*5:]  # [20]
                target_probs = target[i, n, m, self.B*5:]  # [20]
                probs_loss += torch.sum(torch.pow(output_probs - target_probs, 2))
        return probs_loss / bs

    def get_obj_ij_mask(self, output, target, grid_mask_obj):
        bs = grid_mask_obj.size(0)
        obj_ij_mask = torch.zeros((bs, self.S, self.S, self.B))
        obj_ij_ious = torch.zeros((bs, self.S, self.S, self.B))

        for i in range(bs):
            grid_mask_obj_i = grid_mask_obj[i]  # [7, 7]
            de_output = self.decode(output[i])  # [7, 7, 30]
            de_target = self.decode(target[i])  # [7, 7, 30]
            obj_indices = torch.nonzero(grid_mask_obj_i != 0)
            for nm in obj_indices:
                n, m = nm[0], nm[1]
                output_xywh = torch.stack([de_output[n, m, 5*t:5*t+4] for t in range(self.B)], 0)  # [2, 4]
                target_xywh = torch.stack([de_target[n, m, 5*t:5*t+4] for t in range(self.B)], 0)  # [2, 4]
                output_xyxy = YOLOV1LossV1.xywh2xyxy(output_xywh)
                target_xyxy = YOLOV1LossV1.xywh2xyxy(target_xywh)
                ious = YOLOV1LossV1.box_iou(output_xyxy, target_xyxy)[:, 0]
                # 找出哪个box负责
                max_iou_value, max_iou_id = torch.max(ious), torch.argmax(ious)
                obj_ij_mask[i, n, m, max_iou_id] = 1
                obj_ij_ious[i, n, m, max_iou_id] = max_iou_value
        return obj_ij_mask, obj_ij_ious

    def decode(self, tensor):
        # tensor 7 * 7 * 30
        de_tensor = tensor.clone()
        cell_size = 1.0 / float(self.S)
        yv, xv = torch.meshgrid([torch.arange(self.S), torch.arange(self.S)])  # , indexing="ij"
        grids = torch.stack((xv, yv), 2)
        grids_x0y0 = grids * cell_size
        for k in range(self.B):
            s = 5 * k
            xy_normalized = tensor[..., s:s + 2] * cell_size
            xy = xy_normalized.cuda() + grids_x0y0.cuda()
            de_tensor[..., s:s + 2] = xy
            de_tensor[..., s:s + 4] *= self.img_size
        return de_tensor

    @staticmethod
    def box_iou(box1, box2):
        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])
        area1 = box_area(box1.T)
        area2 = box_area(box2.T)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

    @staticmethod
    def xywh2xyxy(boxes):
        boxes_ = torch.zeros(boxes.shape) if isinstance(boxes, torch.Tensor) else np.zeros(boxes.shape)
        boxes_[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return boxes_


if __name__ == '__main__':
    import torch

    target = torch.tensor(np.load(r'demo_data/target.npy'))
    gird_mask_obj = torch.tensor(np.load(r'demo_data/grid_mask_obj.npy'))
    output = torch.tensor(np.load(r'demo_data/output.npy'))

    yolo_loss = YOLOV1LossV1(5, 0.5, 7, 2, 448)

    target = target.unsqueeze(0)
    gird_mask_obj = gird_mask_obj.unsqueeze(0)

    loss = yolo_loss(output, target, gird_mask_obj)
    print(loss)
    print(gird_mask_obj)

