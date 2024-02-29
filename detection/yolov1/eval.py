import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


def eval_net(net, val_loader, criterion, loss_version):
    val_loss = 0
    net.eval()
    # for data in tqdm(val_loader):
    for data in val_loader:
        img_data, target, grid_mask_obj = data[0].cuda(), data[1].cuda(), data[2].cuda()
        output = net(img_data)
        if loss_version == 'v1':
            loss = criterion(output, target, grid_mask_obj)
        else:
            loss = criterion(output, target)
        val_loss += loss.item()
    val_loss_mean = val_loss/len(val_loader)
    return val_loss_mean
