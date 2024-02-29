import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class YOLOV1LossV3(nn.Module):

    def __init__(self, feature_size=7, num_bboxes=2, num_classes=20, lambda_coord=5.0, lambda_noobj=0.5):
        """ Constructor.
        Args:
            feature_size: (int) size of input feature map.
            num_bboxes: (int) number of bboxes per each cell.
            num_classes: (int) number of the object classes.
            lambda_coord: (float) weight for bbox location/size losses.
            lambda_noobj: (float) weight for no-objectness loss.
        """
        super(YOLOV1LossV3, self).__init__()

        self.S = feature_size
        self.B = num_bboxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def compute_iou(self, bbox1, bbox2):
        """ Compute the IoU (Intersection over Union) of two set of bboxes, each bbox format: [x1, y1, x2, y2].
        Args:
            bbox1: (Tensor) bounding bboxes, sized [N, 4].
            bbox2: (Tensor) bounding bboxes, sized [M, 4].
        Returns:
            (Tensor) IoU, sized [N, M].
        """
        N = bbox1.size(0)
        M = bbox2.size(0)

        # Compute left-top coordinate of the intersections
        lt = torch.max(
            bbox1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, :2].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Conpute right-bottom coordinate of the intersections
        rb = torch.min(
            bbox1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Compute area of the intersections from the coordinates
        wh = rb - lt  # width and height of the intersection, [N, M, 2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

        # Compute area of the bboxes
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])  # [N, ]
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])  # [M, ]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N, ] -> [N, 1] -> [N, M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M, ] -> [1, M] -> [N, M]

        # Compute IoU from the areas
        union = area1 + area2 - inter  # [N, M, 2]
        iou = inter / union  # [N, M, 2]

        return iou

    def forward(self, pred_tensor, target_tensor):  # target_tensor[2,0,0,:]
        """ Compute loss for YOLO training. #
        Args:
            pred_tensor: (Tensor) predictions, sized [n_batch, S, S, Bx5+C], 5=len([x, y, w, h, conf]).
            target_tensor: (Tensor) targets, sized [n_batch, S, S, Bx5+C].
        Returns:
            (Tensor): loss, sized [1, ].
        """
        # TODO: Romove redundant dimensions for some Tensors.
        # 获取网格参数S=7,每个网格预测的边框数目B=2，和分类数C=20
        S, B, C = self.S, self.B, self.C
        N = 5 * B + C  # 5=len([x, y, w, h, conf]，N=30

        # 批的大小
        batch_size = pred_tensor.size(0)
        # 有目标的张量[n_batch, S, S]
        coord_mask = target_tensor[..., 4] > 0  # 三个点自动判断维度 自动找到最后一维 用4找出第五个 也就是置信度，为什么30维 第二个框是怎么样的 等下再看
        # 没有目标的张量[n_batch, S, S]
        noobj_mask = target_tensor[..., 4] == 0
        # 扩展维度的布尔值相同，[n_batch, S, S] -> [n_batch, S, S, N]
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target_tensor)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target_tensor)

        # int8-->bool
        noobj_mask = noobj_mask.bool()  # 不是已经bool了？
        coord_mask = coord_mask.bool()

        ##################################################
        # 预测值里含有目标的张量取出来，[n_coord, N]
        coord_pred = pred_tensor[coord_mask].view(-1, N)

        # 提取bbox和C,[n_coord x B, 5=len([x, y, w, h, conf])]
        bbox_pred = coord_pred[:, :5 * B].contiguous().view(-1, 5)  # 防止内存不连续报错
        # 预测值的分类信息[n_coord, C]
        class_pred = coord_pred[:, 5 * B:]

        # 含有目标的标签张量，[n_coord, N]
        coord_target = target_tensor[coord_mask].view(-1, N)

        # 提取标签bbox和C,[n_coord x B, 5=len([x, y, w, h, conf])]
        bbox_target = coord_target[:, :5 * B].contiguous().view(-1, 5)
        # 标签的分类信息
        class_target = coord_target[:, 5 * B:]
        ######################################################

        # ##################################################
        # 没有目标的处理
        # 找到预测值里没有目标的网格张量[n_noobj, N]，n_noobj=SxS-n_coord
        noobj_pred = pred_tensor[noobj_mask].view(-1, N)
        # 标签的没有目标的网格张量 [n_noobj, N]
        noobj_target = target_tensor[noobj_mask].view(-1, N)

        noobj_conf_mask = torch.cuda.BoolTensor(noobj_pred.size()).fill_(0)  # [n_noobj, N]
        for b in range(B):
            noobj_conf_mask[:,
            4 + b * 5] = 1  # 没有目标置信度置1，noobj_conf_mask[:, 4] = 1; noobj_conf_mask[:, 9] = 1 目标是下面把置信度拿出来再并排

        noobj_pred_conf = noobj_pred[noobj_conf_mask]  # [n_noobj x 2=len([conf1, conf2])] 这里目标是
        noobj_target_conf = noobj_target[noobj_conf_mask]  # [n_noobj x 2=len([conf1, conf2])]
        # 计算没有目标的置信度损失 加法》？ #如果 reduction 参数未指定，默认值为 'mean'，表示对所有元素的误差求平均值。
        # loss_noobj=F.mse_loss(noobj_pred_conf, noobj_target_conf,)*len(noobj_pred_conf)
        loss_noobj = F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction='sum')
        #################################################################################

        #################################################################################
        # Compute loss for the cells with objects.
        coord_response_mask = torch.cuda.BoolTensor(bbox_target.size()).fill_(0)  # [n_coord x B, 5]
        coord_not_response_mask = torch.cuda.BoolTensor(bbox_target.size()).fill_(1)  # [n_coord x B, 5]
        bbox_target_iou = torch.zeros(bbox_target.size()).cuda()  # [n_coord x B, 5], only the last 1=(conf,) is used

        # Choose the predicted bbox having the highest IoU for each target bbox.
        for i in range(0, bbox_target.size(0), B):
            pred = bbox_pred[i:i + B]  # predicted bboxes at i-th cell, [B, 5=len([x, y, w, h, conf])]
            pred_xyxy = Variable(torch.FloatTensor(pred.size()))  # [B, 5=len([x1, y1, x2, y2, conf])]
            # Because (center_x,center_y)=pred[:, 2] and (w,h)=pred[:,2:4] are normalized for cell-size and image-size respectively,
            # rescale (center_x,center_y) for the image-size to compute IoU correctly.
            pred_xyxy[:, :2] = pred[:, :2] / float(S) - 0.5 * pred[:, 2:4]
            pred_xyxy[:, 2:4] = pred[:, :2] / float(S) + 0.5 * pred[:, 2:4]

            target = bbox_target[
                i]  # target bbox at i-th cell. Because target boxes contained by each cell are identical in current implementation, enough to extract the first one.
            target = bbox_target[i].view(-1, 5)  # target bbox at i-th cell, [1, 5=len([x, y, w, h, conf])]
            target_xyxy = Variable(torch.FloatTensor(target.size()))  # [1, 5=len([x1, y1, x2, y2, conf])]
            # Because (center_x,center_y)=target[:, 2] and (w,h)=target[:,2:4] are normalized for cell-size and image-size respectively,
            # rescale (center_x,center_y) for the image-size to compute IoU correctly.
            target_xyxy[:, :2] = target[:, :2] / float(S) - 0.5 * target[:, 2:4]
            target_xyxy[:, 2:4] = target[:, :2] / float(S) + 0.5 * target[:, 2:4]

            iou = self.compute_iou(pred_xyxy[:, :4], target_xyxy[:, :4])  # [B, 1]
            max_iou, max_index = iou.max(0)
            max_index = max_index.data.cuda()

            coord_response_mask[i + max_index] = 1
            coord_not_response_mask[i + max_index] = 0

            # "we want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth"
            # from the original paper of YOLO.
            bbox_target_iou[i + max_index, torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
        bbox_target_iou = Variable(bbox_target_iou).cuda()

        # BBox location/size and objectness loss for the response bboxes.
        bbox_pred_response = bbox_pred[coord_response_mask].view(-1, 5)  # [n_response, 5]
        bbox_target_response = bbox_target[coord_response_mask].view(-1,
                                                                     5)  # [n_response, 5], only the first 4=(x, y, w, h) are used
        target_iou = bbox_target_iou[coord_response_mask].view(-1,
                                                               5)  # [n_response, 5], only the last 1=(conf,) is used
        loss_xy = F.mse_loss(bbox_pred_response[:, :2], bbox_target_response[:, :2], reduction='sum')
        loss_wh = F.mse_loss(torch.sqrt(bbox_pred_response[:, 2:4]), torch.sqrt(bbox_target_response[:, 2:4]),
                             reduction='sum')
        loss_obj = F.mse_loss(bbox_pred_response[:, 4], target_iou[:, 4], reduction='sum')

        ################################################################################

        # Class probability loss for the cells which contain objects.
        loss_class = F.mse_loss(class_pred, class_target, reduction='sum')

        print('regression_loss_obj', self.lambda_coord * (loss_xy + loss_wh))
        print('confidence_loss_obj', loss_obj)
        print('confidence_loss_noobj', self.lambda_noobj * loss_noobj)
        print('classify_loss_obj', loss_class)

        # Total loss
        loss = self.lambda_coord * (loss_xy + loss_wh) + loss_obj + self.lambda_noobj * loss_noobj + loss_class
        loss = loss / float(batch_size)

        return loss

if __name__ == '__main__':
    import torch
    import numpy as np
    target = torch.tensor(np.load(r'demo_data/target.npy'))
    gird_mask_obj = torch.tensor(np.load(r'demo_data/grid_mask_obj.npy'))
    output = torch.tensor(np.load(r'demo_data/output.npy'))

    yolo_loss = YOLOV1LossV3(feature_size=7, num_bboxes=2, num_classes=6, lambda_coord=5.0, lambda_noobj=0.5)

    target = target.unsqueeze(0)
    gird_mask_obj = gird_mask_obj.unsqueeze(0)

    # print(output.shape, target.shape, gird_mask_obj.shape)
    loss = yolo_loss(output.cuda(), target.cuda())
    print(loss)
    print(gird_mask_obj)