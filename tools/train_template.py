import argparse
import os
import csv
import sys
import wandb
import yaml
import ruamel.yaml
from tqdm import tqdm

import logging
import time
import pathlib
import numpy as np
import cv2
import psutil
from PIL import Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda import amp
from torch.utils.data import Dataset,DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms.functional as TF

import _init_paths as _init_paths
import models, datasets
from core.function import validate
from utils.utils import FullModel
from utils.utils import ModelEmaV2, select_device, torch_distributed_zero_first,init_seeds,set_logging,createCheckpoint
import utils.losses as loss_funcs
import utils.metrics as metrics

from datasets.dataset import get_dataset
from test import call_inference

import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold

import kindle
from kindle import Model, TorchTrainer

logger = logging.getLogger(__name__)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Нужна кастомная функция
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# def get_x_y(filename):
#     img_files = []
#     mask_files = []

#     filenames = []
#     with open(filename, newline='') as f:
#         reader = csv.reader(f)
#         filenames = list(reader)

#     for i in range(len(filenames)):
#         file = filenames[i][0]
#         img_files.append('./data/' + file)
#         mask_name = file.replace('/img/', '/masks/')
#         mask_files.append('./data/' + mask_name)

#     return np.asarray(img_files), np.asarray(mask_files)

def resume(model, path):
    pretrained_state = torch.load(path, map_location='cpu')
    optimizer = pretrained_state['optimizer_state_dict']
    scheduler = pretrained_state['scheduler']
    start_epoch = int(pretrained_state['last_epoch'])

    model_dict = model.state_dict()
    model_dict.update(pretrained_state['model_state_dict'])
    model.load_state_dict(model_dict, strict = False)

    return model, optimizer, scheduler, start_epoch

def load_config(hyp):
    with open(hyp) as f:
        experiment_dict = yaml.load(f, Loader=yaml.FullLoader)

    return experiment_dict

def run_model(hyp,opt,device,wandb):
    batch_size,total_batch_size =  opt.batch_size,opt.total_batch_size
    rank = opt.global_rank
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)

    experiment_dict = load_config(hyp)
    model = Model("./tools/kindle/temp.yaml",verbose=False).to(device) #init model

    if opt.inference:
        call_inference(experiment_dict, model, device)
        return

    X_train, y_train = get_x_y(experiment_dict['dataset']['train_set'])
    X_test, y_test = get_x_y(experiment_dict['dataset']['test_set'])

    dataset, dataloader = get_dataset(experiment_dict, X_train, y_train, batch_size, world_size=opt.world_size, rank=rank)
    testset, testloader = get_dataset(experiment_dict, X_test, y_test, batch_size*2, world_size=opt.world_size, rank=rank, val=True)

    loss_function = loss_funcs.create_loss_function(experiment_dict)

    start_epoch, best_mIoU = 0, 0.0
    if experiment_dict['train']['resume']:
        model, optimizer_state_dict, scheduler_state_dict, start_epoch = resume(model, experiment_dict['resumepath']) #load weights
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    nbs = experiment_dict['train']['accumulate_batch_size']
    accumulate = max(round(nbs / total_batch_size), 1)
    experiment_dict['train']['weight_decay'] *= total_batch_size * accumulate / nbs

    if experiment_dict['train']['optimizer'] == 'SGD':
        params_dict = dict(model.named_parameters())
        params = [{'params': list(params_dict.values()), 'lr': experiment_dict['train']['lr']}]
        optimizer = torch.optim.SGD(params,
                                lr=experiment_dict['train']['lr'],
                                momentum=experiment_dict['train']['momentum'],
                                weight_decay=experiment_dict['train']['weight_decay'],
                                nesterov=experiment_dict['train']['nesterov'])

        if experiment_dict['train']['resume']:
            optimizer.load_state_dict(optimizer_state_dict)

    if experiment_dict['train']['scheduler'] == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer,step_size=experiment_dict['train']['step_size'],
                                                  gamma=experiment_dict['train']['gamma'])
        if experiment_dict['train']['resume']:
            scheduler.load_state_dict(scheduler_state_dict)

    if experiment_dict['train']['ema']==True:
        ema = ModelEmaV2(model,decay=experiment_dict['train']['ema_decay']) if rank in [-1, 0] else None

    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()

    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    if wandb and wandb.run is None:
        experiment_dict['batch_size']=batch_size
        experiment_dict['total_batch_size']=total_batch_size
        experiment_dict['workers']=opt.workers
        experiment_dict['sync_bn']=opt.sync_bn
        wandb_run = wandb.init(config=experiment_dict, resume=experiment_dict['train']['resume'],
                               project=experiment_dict['project'],
                               name=experiment_dict['experiment_name'],
                               id=None)

    nb = len(dataloader)  # number of batches
    if experiment_dict['train']['ema']==True:
        ema.updates = start_epoch * nb // accumulate  # set EMA updates

    scheduler.last_epoch = start_epoch - 1
    scaler = amp.GradScaler(enabled=cuda)
    epochs = experiment_dict['train']['epochs']
    metricsFactory = metrics.Metrics_factory()

    print("start_epoch: ", start_epoch)

    for epoch in range(start_epoch, epochs):
        cv2.setNumThreads(2)

        mean_train_loss = 0
        mean_train_acc = 0
        mean_train_f1 = 0
        mean_train_bIoU = 0

        model.train()
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()

#        with torch.profiler.profile(schedule=torch.profiler.schedule(
#                                    wait=2,
#                                    warmup=2,
#                                    active=6,
#                                    repeat=1),
#                                    with_stack=True) as profiler:
        for i, inp in pbar:
            ni = i + nb * epoch
            imgs = inp[0].cuda()
            labels = inp[1].long().cuda()

            labels_onehot = F.one_hot(labels, num_classes=3).permute((0,3,1,2))

            with amp.autocast(enabled=cuda):
                # start_time = time.time()
                outputs = model(imgs)
                # torch.cuda.synchronize()
                # time_taken = time.time() - start_time
                # print(time_taken)

                outputs_softmax = [F.softmax(outputs[0][0], dim = 1), F.softmax(outputs[0][1], dim = 1)]

                # start_time = time.time()
                losses = loss_funcs.compute_loss(loss_function, outputs_softmax, labels_onehot)
                # torch.cuda.synchronize()
                # time_taken = time.time() - start_time
                # print(time_taken)

            if rank != -1:
                losses *= opt.world_size

            loss = 0
            for j in range(len(losses)):
                loss += experiment_dict['train']['loss_weights'][j]*losses[j]

            # print('Loss:', loss.item(), '\nCE Loss:', losses[0].item(), '\nDice Loss:', losses[1].item(), '\n')
            scaler.scale(loss).backward()

            if ni % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
#                   profiler.step()
                if experiment_dict['train']['ema']==True:
                    ema.update(model)
            acc, f1 = metricsFactory.compute_metrics(outputs_softmax[0], labels, False)
            acc = acc.mean()
            f1 = f1.mean()

            if rank in [-1, 0]:
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = 'Epoch: ' + str(epoch) +'/'+str(epochs - 1) +' Memory: '+str(mem) +' Train Loss: '+str(loss.item())
                mean_train_loss+=loss.item()
                mean_train_acc+=acc.item()
                mean_train_f1+=f1.item()

        scheduler.step()

        if experiment_dict['validate']:
            cv2.setNumThreads(8)
            val_loss, mIoU, IoU_arr, bIoU_arr, val_acc, val_f1, bIoU = validate(experiment_dict, testloader, model, None, loss_function)
            #print('AAAAAAAAAAAAAAAAAAAAA', bIoU_arr)
            if mIoU >= best_mIoU:
                best_mIoU = mIoU

            createCheckpoint(experiment_dict['savepath'],model,optimizer,epoch,best_mIoU,scheduler)

            if wandb:
                results_dict={}
                results_dict['train_loss']=mean_train_loss/len(dataloader)
                results_dict['val_loss'] = val_loss
                results_dict['rail_IoU'] = IoU_arr[1]
                results_dict['wagon_IoU'] = IoU_arr[2]
                results_dict['mean_IoU'] = mIoU
                results_dict['train_accuracy']=mean_train_acc/len(dataloader)
                results_dict['test_accuracy']=val_acc
                results_dict['train_f1'] = mean_train_f1/len(dataloader)
                results_dict['test_f1'] = val_f1
                results_dict['rail_bIoU'] = bIoU_arr[0]
                results_dict['wagon_bIoU'] = bIoU_arr[1]
                results_dict['test_bIoU'] = bIoU
                wandb.log(results_dict)

            else:
                best_mIoU = 0.0
                createCheckpoint(experiment_dict['savepath'],model,optimizer,epoch,best_mIoU,scheduler)

            final_epoch = epoch + 1 == epochs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str,
                        default='./lib/config/train_config.yaml',
                        help='hyperparameters path')
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=64, help='maximum number of dataloader workers')
    parser.add_argument('--inference', type=bool, default=False, help='inference flag')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    opt = parser.parse_args()
    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)

    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size
    logger.info(opt)

    run_model(opt.hyp,opt,device,wandb)
