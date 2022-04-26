import logging
import os
import time
import cv2

import numpy as np
import numpy.ma as ma
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda import amp

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate
import utils.losses as loss_funcs
import utils.metrics as metrics

import utils.distributed as dist

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size

def validate(config, testloader, model, loss_function):
    model.eval()
    ave_loss = AverageMeter()
    nums = config['model']['num_outputs']
    confusion_matrix = np.zeros(
        (config['model']['num_classes'], config['model']['num_classes'], nums))
    metricsFactory = metrics.Metrics_factory()
    mean_f1 = []
    mean_acc = []
    mean_bIoU = []
    bIoU_array = []
    ct = 0
    out = None
    time_check = False
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(testloader)):
            image, label, items = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            label[label >= 1] -= 1
            labels_onehot = F.one_hot(label, num_classes=3).permute((0,3,1,2))

            if not time_check:
                start.record()
                pred = model(image)
                end.record()
                torch.cuda.synchronize()
                print('Time: ', start.elapsed_time(end))
                time_check = True
            else:
                pred = model(image)

            pred_softmax = [F.softmax(pred[0][0], dim = 1), F.softmax(pred[0][1], dim = 1)]

            if idx == 0:
                out = []
                for file in items:
                    true_img = cv2.imread(file)
                    true_img = cv2.resize(true_img, (config['dataset']['img_size'][1], config['dataset']['img_size'][0]))
                    out.append(true_img)
                out = np.uint8(np.asarray(out))
                for j in range(len(out)):
                    _, result = torch.max(pred_softmax[0][j], dim=0)
                    overlay = np.uint8(F.one_hot(result, num_classes=3).cpu().numpy() * 255)
                    out[j] = cv2.addWeighted(out[j],0.7,overlay,0.3,0)

            losses = loss_funcs.compute_loss(loss_function, pred_softmax, pred[0][:2], labels_onehot)

            isw_loss = pred[1]
            pred = torch.softmax(pred[0][0], dim = 1)

            acc, f1, b_IoU, b_IoU_arr = metricsFactory.compute_metrics(pred_softmax[0], label)

            confusion_matrix[..., 0] += get_confusion_matrix(label,
                                                             pred,
                                                             size,
                                                             config['model']['num_classes'],
                                                             config['train']['ignore_label'])

            loss = 0
            loss = sum(losses) + 0.6*isw_loss

            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss

            ave_loss.update(reduced_loss.item())
            mean_acc+=acc.tolist()
            mean_f1+=f1.tolist()
            mean_bIoU+=list(b_IoU)
            bIoU_array.append(b_IoU_arr)


    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(1):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array[1:].mean()
        mean_IoU_all = IoU_array.mean()

    bIoU_array = np.array(bIoU_array)
    mean_bIoU_array = np.mean(bIoU_array, axis=(1,2))

    return ave_loss.average(), mean_IoU, mean_IoU_all, IoU_array, mean_bIoU_array, np.mean(mean_acc), np.mean(mean_f1), np.mean(mean_bIoU), out
