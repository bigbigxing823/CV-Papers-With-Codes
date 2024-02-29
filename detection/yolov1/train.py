import os
import argparse
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from dataset import YOLODataset, get_train_transforms, get_valid_transforms
from models.resnet_yolo import resnet50
from models.yolo import YoloV1
from torchvision import models
from losses.loss_v1 import YOLOV1LossV1
from losses.loss_v2 import YOLOV1LossV2
from losses.loss_v3 import YOLOV1LossV3
from eval import eval_net


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_net(args):
    # 类别
    classes_txt = open(args.classes_txt_path, 'r')
    names = [name.strip() for name in classes_txt.readlines()]

    # 定义模型
    if args.net == 'yolo':
        net = YoloV1(args.img_size, args.in_channels, args.num_classes, args.grid_num, args.box_num)
        net.cuda()
    elif args.net == 'res_yolo':
        net = resnet50()
        resnet = models.resnet50(pretrained=True)
        new_state_dict = resnet.state_dict()
        dd = net.state_dict()
        for k in new_state_dict.keys():
            if k in dd.keys() and not k.startswith('fc'):
                dd[k] = new_state_dict[k]
        net.load_state_dict(dd)
        net.cuda()

    # 构建损失
    if args.loss_version == 'v1':
        criterion = YOLOV1LossV1(5, 0.5, 7, 2, 448)
    elif args.loss_version == 'v2':
        criterion = YOLOV1LossV2(7, 2, 6, 5, 0.5)
    else:
        criterion = YOLOV1LossV3(7, 2, 6, 5.0, 0.5)

    # 构建优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # 保存信息
    save_info = {}
    save_info['state_dict'] = net.state_dict()
    save_info['img_size'] = args.img_size
    save_info['in_channels'] = args.in_channels
    save_info['num_classes'] = args.num_classes
    save_info['grid_num'] = args.grid_num
    save_info['box_num'] = args.box_num
    save_info['names'] = names

    # 数据迭代器
    train_dataset = YOLODataset(args.train_txt_path,
                                S=args.grid_num,
                                B=args.box_num,
                                C=args.num_classes,
                                transforms=get_train_transforms(),
                                train=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4)

    # 评估
    val_dataset = YOLODataset(args.val_txt_path,
                              S=args.grid_num,
                              B=args.box_num,
                              C=args.num_classes,
                              transforms=get_valid_transforms(),
                              train=False)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=4)

    # 开始训练
    train_losses = []
    val_losses = []
    min_val_loss = float('inf')
    learning_rate = args.lr
    num_iter = 0
    for i in range(args.epoch):
        print('Epoch:' + str(i))

        # 学习率调整
        if i == 80:
            learning_rate = 0.0001
        if i == 150:
            learning_rate = 0.00001
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        # 迭代训练
        net.train()
        step_losses = 0
        for step, data in enumerate(train_loader):
            img_data, target, grid_mask_obj = data[0].cuda(), data[1].cuda(), data[2].cuda()
            output = net(img_data)
            if args.loss_version == 'v1':
                loss = criterion(output, target, grid_mask_obj)
            else:
                loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_losses += loss.item()

            if (step+1) % 5 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f'
                %(i+1, args.epoch, step+1, len(train_loader), loss.item(), step_losses / (step+1)))
                num_iter += 1

        epoch_loss_epoch = step_losses/len(train_loader)
        train_losses.append(epoch_loss_epoch)

        # 模型验证
        with torch.no_grad():
            val_loss_epoch = eval_net(net, val_loader, criterion, args.loss_version)
            val_losses.append(val_loss_epoch)

            if val_loss_epoch < min_val_loss:
                min_val_loss = val_loss_epoch
                torch.save(save_info, os.path.join(args.save_dir, 'yolov1.pt'))
                print('get best val loss %.5f' % min_val_loss)

            print('train loss:', '{}'.format('%.3f' % (epoch_loss_epoch)), 'val loss:', val_loss_epoch)

    # 绘制损失函数曲线
    epoch_np = np.array(list(range(len(train_losses))))
    train_losses_np = np.array(train_losses)
    val_losses_np = np.array(val_losses)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('损失变化曲线')
    plt.plot(epoch_np, train_losses_np, color='green', label='train loss')
    plt.plot(epoch_np, val_losses_np, color='red', label='val loss')
    plt.xlabel('迭代次数')
    plt.ylabel('损失变化')
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, 'loss.png'))


if __name__ == '__main__':
    seed_everything(719)

    # 命令行参数
    parser = argparse.ArgumentParser()

    # 数据
    parser.add_argument('--train_txt_path', type=str, default=r'data/NEU-DET/train.txt', help='path to train.txt')
    parser.add_argument('--val_txt_path', type=str, default=r'data/NEU-DET/val.txt', help='path to val.txt')
    parser.add_argument('--classes_txt_path', type=str, default='data/NEU-DET/classes.txt', help='path to classes.txt')

    # 训练
    parser.add_argument('--epoch', type=int, default=240, help='train epoch')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='save checkpoint path')

    # 模型
    parser.add_argument('--img_size', type=int, default=448, help='image size')
    parser.add_argument('--in_channels', type=int, default=3, help='input channels')
    parser.add_argument('--num_classes', type=int, default=6, help='number of class')
    parser.add_argument('--grid_num', type=int, default=7, help='grad numer of model (S)')
    parser.add_argument('--box_num', type=int, default=2, help='box number of model (B)')
    parser.add_argument('--gamma_coord', type=float, default=5, help='weight of coord')
    parser.add_argument('--gamma_noobj', type=float, default=0.5, help='weight of noobj')
    parser.add_argument('--loss_version', type=str, default='v1', help='version of yolov1 loss')
    parser.add_argument('--net', type=str, default='res_yolo', help='network of yolo')

    args_ = parser.parse_args()
    # 训练
    train_net(args_)
