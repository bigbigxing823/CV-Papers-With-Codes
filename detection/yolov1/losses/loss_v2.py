import torch
from torch.nn import Module, functional
from torch.autograd import Variable


def iou_compute(bbox_1, bbox_2):
    N, M = bbox_1.size(0), bbox_2.size(0)  # [N， 4=(x1, y2, x2, y2)], [M， 4=(x, y, w, y)]

    left_top = torch.max(bbox_1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N, 2] -> [N, 1, 2] -> [N, M, 2]
                         bbox_2[:, :2].unsqueeze(0).expand(N, M, 2))  # [M, 2] -> [1, M, 2] -> [N, M, 2]
    right_bottom = torch.min(bbox_1[:, 2:].unsqueeze(1).expand(N, M, 2),
                             bbox_2[:, 2:].unsqueeze(0).expand(N, M, 2))

    wh = right_bottom - left_top  # 【N, M， 2】
    wh[wh < 0] = 0  # w, h < 0, 说明没有相交区域， 直接设置为 0

    # 求面积 w * h
    inter = wh[:, :, 0] * wh[:, :, 1]  # w * h
    area_1 = (bbox_1[:, 2] - bbox_1[:, 0]) * (bbox_1[:, 3] - bbox_1[:, 1])
    area_2 = (bbox_2[:, 2] - bbox_2[:, 0]) * (bbox_2[:, 3] - bbox_2[:, 1])
    area_1 = area_1.unsqueeze(1).expand_as(inter)
    area_2 = area_2.unsqueeze(0).expand_as(inter)

    return inter / (area_1 + area_2 - inter)  # [N, M, 2], iou


class YOLOV1LossV2(Module):
    def __init__(self, num_grids, num_bboxes, num_classes, i_coord, i_noobj):
        super(YOLOV1LossV2, self).__init__()

        self.S = num_grids
        self.B = num_bboxes
        self.C = num_classes
        self.i_coord = i_coord
        self.i_noobj = i_noobj
        self.N = 5 * num_bboxes + num_classes  # [x, y, w, h, conf] x num_bbox + num_class

    def forward(self, pred_tensor, target_tensor):

        batch_size = pred_tensor.size(0)  # pred_tensor = [batchsize, S, S, N=Bx5+C]

        coord_pred, coord_target = self.get_lambda_i_obj(pred_tensor, target_tensor)

        bbox_pred = coord_pred[:, :5 * self.B].contiguous().view(-1,
                                                                 5)  # 网格含目标的bbox集合， [n_coord x B, 5=(x, y, w, h, conf)]
        bbox_target = coord_target[:, :5 * self.B].contiguous().view(-1, 5)

        coord_response_mask, bbox_target_iou = self.get_lambda_ij_obj(bbox_pred, bbox_target)
        bbox_pred_response = bbox_pred[coord_response_mask].view(-1, 5)  # x
        bbox_target_response = bbox_target[coord_response_mask].view(-1, 5)
        target_iou = bbox_target_iou[coord_response_mask].view(-1, 5)

        noobj_pred_conf, noobj_target_conf = self.get_lambda_ij_noobj(pred_tensor, target_tensor)

        loss_wh = functional.mse_loss(torch.sqrt(bbox_pred_response[:, 2:4]),
                                      torch.sqrt(bbox_target_response[:, 2:4]), reduction='sum')
        loss_xy = functional.mse_loss(bbox_pred_response[:, :2],
                                      bbox_target_response[:, :2], reduction='sum')
        loss_obj = functional.mse_loss(bbox_pred_response[:, 4], target_iou[:, 4], reduction='sum')
        loss_noobj = functional.mse_loss(noobj_pred_conf, noobj_target_conf, reduction='sum')
        loss_class = functional.mse_loss(coord_pred[:, 5 * self.B:],
                                         coord_target[:, 5 * self.B:], reduction='sum')

        # print('regression_loss_obj', self.i_coord * (loss_xy + loss_wh))
        # print('confidence_loss_obj', loss_obj)
        # print('confidence_loss_noobj', self.i_noobj * loss_noobj)
        # print('classify_loss_obj', loss_class)

        total_loss = self.i_coord * (loss_xy + loss_wh) + loss_obj + self.i_noobj * loss_noobj + loss_class
        total_loss = total_loss / float(batch_size)

        return total_loss

    def get_lambda_ij_noobj(self, pred_tensor, target_tensor):
        noobj_mask = target_tensor[..., 4] == 0  # mask=[batchsize, S, S], bool
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target_tensor)  # mask=[batchsize, S, S, N]

        # 网格没有目标的
        noobj_pred, noobj_target = pred_tensor[noobj_mask].view(-1, self.N), target_tensor[noobj_mask].view(-1, self.N)

        noobj_conf_mask = torch.zeros(noobj_pred.size(), dtype=torch.bool).cuda()
        for b in range(self.B):
            noobj_conf_mask[:, 4 + b * 5] = 1  # 将不含目标的bbox的参数conf变成1
        noobj_pred_conf, noobj_target_conf = noobj_pred[noobj_conf_mask], noobj_target[noobj_conf_mask]

        return noobj_pred_conf, noobj_target_conf

    def get_lambda_i_obj(self, pred_tensor, target_tensor):
        coord_mask = target_tensor[..., 4] > 0  # 因为有2个bounding box，这里判断的应该是第一个bbox的conf吧，有进行排序吗？会降低一维度
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target_tensor)  # 在最后一维加一维，整得和tensor一样

        # 相当于已经知道在网格内是否有目标了，但是不知道具体是哪个bbox是负责目标的 I_i^{obj}
        coord_pred = pred_tensor[coord_mask].view(-1, self.N)
        coord_target = target_tensor[coord_mask].view(-1, self.N)
        return coord_pred, coord_target

    def get_lambda_ij_obj(self, bbox_pred, bbox_target):
        # 通过bbox计算出iou来确定某个网格的某个bbox是否为一个目标负责
        # buffer
        bbox_with_obj_mask = torch.zeros(bbox_target.size(0), dtype=torch.bool).cuda()
        bbox_without_obj_mask = torch.ones(bbox_target.size(0), dtype=torch.bool).cuda()
        bbox_target_iou = torch.zeros(bbox_target.size()).cuda()
        # 遍历网格内的bbox
        for i in range(0, bbox_target.size(0), self.B):
            # 预测值与真实值的坐标重新转换（由于归一化）
            pred_xyxy = self.denormalize(bbox_pred[i: i + self.B])
            target_xyxy = self.denormalize(bbox_target[i].view(-1, 5))
            # max iou (ground truth box and bbox)
            iou = iou_compute(pred_xyxy, target_xyxy)
            max_iou, max_index = iou.max(0)

            max_index = max_index.data.cuda()
            bbox_with_obj_mask[i + max_index] = 1  # 这个就是 i_{ij}^{obj}
            bbox_without_obj_mask[i + max_index] = 0  # 好像没有用到

            bbox_target_iou[i + max_index, torch.LongTensor([4]).cuda()] = max_iou.data.cuda()  # 只填充conf的位置
        bbox_target_iou = Variable(bbox_target_iou).cuda()

        return bbox_with_obj_mask, bbox_target_iou

    def denormalize(self, xywh):
        # xywh 转成 xyxy 格式，同时反归一化，恢复原来尺寸
        xyxy = Variable(torch.FloatTensor(xywh.size()))
        xyxy[:, :2] = xywh[:, :2] / float(self.S) - 0.5 * xywh[:, 2:4]
        xyxy[:, 2:4] = xywh[:, :2] / float(self.S) + 0.5 * xywh[:, 2:4]
        return xyxy[:, :4]


if __name__ == '__main__':
    import torch
    import numpy as np

    target = torch.tensor(np.load(r'demo_data/target.npy')).cuda()
    gird_mask_obj = torch.tensor(
        np.load(r'demo_data/grid_mask_obj.npy')).cuda()
    output = torch.tensor(np.load(r'demo_data/output.npy')).cuda()
    target = target.unsqueeze(0).cuda()
    loss = YOLOV1LossV2(7, 2, 6, 5, 0.5)
    j = loss.forward(output, target)
    print(j)
