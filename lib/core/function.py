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

def validate(config, testloader, model, writer_dict, loss_function):
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
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(testloader)):
            image, label, _, names = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            label[label >= 1] -= 1
            labels_onehot = F.one_hot(label, num_classes=3).permute((0,3,1,2))

            # start_time = time.time()
            pred = model(image)
            # torch.cuda.synchronize()
            # time_taken = time.time() - start_time
            # print(time_taken)

            pred_softmax = [F.softmax(pred[0][0], dim = 1), F.softmax(pred[0][1], dim = 1)]

            # start_time = time.time()
            losses = loss_funcs.compute_loss(loss_function, pred_softmax, labels_onehot)
            # torch.cuda.synchronize()
            # time_taken = time.time() - start_time
            # print(time_taken)

            isw_loss = pred[1]
            pred = [torch.softmax(pred[0][0], dim = 1)]

            # start_time = time.time()
            acc, f1, b_IoU, b_IoU_arr = metricsFactory.compute_metrics(pred_softmax[0], label)
            # torch.cuda.synchronize()
            # time_taken = time.time() - start_time
            # print(time_taken)


            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config['model']['align_corners']
                )

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config['model']['num_classes'],
                    config['train']['ignore_label']
                )

            loss = 0
            loss_coeffs = [0.7, 0.3]
            for j in range(len(losses)):
                loss = loss + loss_coeffs[j]*losses[j]
            loss = loss + 0.6*isw_loss

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
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))

    bIoU_array = np.array(bIoU_array)
    #mean_bIoU_array = []
    print(bIoU_array.shape)
    mean_bIoU_array = np.mean(bIoU_array, axis=(1,2))
    #mean_bIoU_array = [(np.sum(bIoU_array[...,0]) + np.sum(bIoU_array[...,0] == -1))/(len(bIoU_array[...,0]) - np.sum(bIoU_array[...,0] == -1)),
    #                   (np.sum(bIoU_array[...,1]) + np.sum(bIoU_array[...,1] == -1))/(len(bIoU_array[...,1]) - np.sum(bIoU_array[...,0] == -1))]
    return ave_loss.average(), mean_IoU, IoU_array, mean_bIoU_array, np.mean(mean_acc), np.mean(mean_f1), np.mean(mean_bIoU)
