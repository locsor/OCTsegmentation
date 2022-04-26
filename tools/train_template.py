import argparse
import os, ast
import csv
import sys, json
import wandb
import yaml
import ruamel.yaml
from tqdm import tqdm
import pandas as pd

import logging
import time
import pathlib
import numpy as np
import cv2
import psutil
from PIL import Image

import torch
import torch.nn as nn
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

def get_x_y(filename):
    df = pd.read_csv(filename, index_col = False, sep='\t')
    imgs = df['img'].tolist()
    masks = df['mask'].tolist()

    for i in range(len(imgs)):
        if '534_train_1908150000.zcm.0_source_data/img/1565860783000000' in imgs[i]: #broken folder
            continue

        if ':' in imgs[i]:
            imgs[i] = imgs[i].replace(':', '')
            masks[i] = masks[i].replace(':', '')

        if os.path.isfile(masks[i]) == False:
            # print(masks[i])
            masks[i] = masks[i].replace('/masks_machine/', '/masks/')

    return imgs, masks


def resume(model, path, experiment_dict, X, y):
    state_pre_isw = torch.load('./pretrained_models/ddrnet_model_4.pt')
    model_dict_pre_isw = model.state_dict()
    model_dict_pre_isw.update(state_pre_isw['model_state_dict'])
    model.load_state_dict(model_dict_pre_isw, strict = False)
    model = init_iws(model, experiment_dict, X, y)

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

    yaml_ruamel = ruamel.yaml.YAML()
    with open(experiment_dict['kindle_model_path']) as fp:
        data = yaml_ruamel.load(fp)

    data["backbone"][0][-1][-1] = experiment_dict['robust']

    with open("./tools/kindle/temp.yaml", 'w') as yaml_file:
        yaml_ruamel.dump(data, yaml_file)

    return experiment_dict

def init_iws( model, experiment_dict, X, y):

    dataset, dataloader = get_dataset(experiment_dict, X, y, batch_size=1, world_size=opt.world_size, rank=-1, set_var=True, val=True)

    nb = len(dataloader)  # number of batches
    pbar = tqdm(enumerate(dataloader), total=nb)

    model.eval()
    with torch.no_grad():
        for i, inp in pbar:
            imgs = torch.cat(inp[0], axis=0).float().cuda()
            _ = model(imgs)

    return model

def run_model(hyp,opt,device,wandb):
    cv2.setNumThreads(4)
    torch.set_num_threads(4)
    batch_size,total_batch_size =  opt.batch_size,opt.total_batch_size
    rank = opt.global_rank
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)

    experiment_dict = load_config(hyp)
    model = Model("./tools/kindle/temp.yaml",verbose=False).to(device) #init model

    X_train, y_train = get_x_y(experiment_dict['dataset']['train_set'])
    X_test, y_test = get_x_y(experiment_dict['dataset']['test_set'])


    dataset, dataloader = get_dataset(experiment_dict, X_train, y_train, batch_size, 
                                      world_size=opt.world_size, workers=opt.workers, rank=rank, cuda=cuda)
    testset, testloader = get_dataset(experiment_dict, X_test, y_test, 2*batch_size,
                                      world_size=opt.world_size, workers=opt.workers, rank=rank, val=True, cuda=cuda)

    loss_function = loss_funcs.create_loss_function(experiment_dict, device)

    start_epoch, best_mIoU = 0, 0.0
    if experiment_dict['train']['resume']:
        model, optimizer_state_dict, scheduler_state_dict, start_epoch = resume(model, experiment_dict['resumepath'], experiment_dict, X_test, y_test) #load weights
    else:
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    if opt.validate:
        val_loss, mIoU, mIoU_all, IoU_arr, bIoU_arr, val_acc, val_f1, bIoU, out_images = validate(experiment_dict, testloader, model, loss_function)

    else:
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
            # scheduler = lr_scheduler.StepLR(optimizer,step_size=experiment_dict['train']['step_size'],
                                                      # gamma=experiment_dict['train']['gamma'])

            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=experiment_dict['train']['max_OneCycleLR'], 
                                                steps_per_epoch=len(dataloader), epochs=60,
                                                div_factor=10, final_div_factor=1e4)
            if experiment_dict['train']['resume']:
                scheduler.load_state_dict(scheduler_state_dict)

        if experiment_dict['train']['ema']==True:
            ema = ModelEmaV2(model,decay=experiment_dict['train']['ema_decay']) if rank in [-1, 0] else None

        if wandb and wandb.run is None:
            experiment_dict['batch_size']=batch_size
            experiment_dict['total_batch_size']=total_batch_size
            experiment_dict['workers']=opt.workers
            # experiment_dict['sync_bn']=opt.sync_bn
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

        weightingLoss = loss_funcs.CoVWeightingLoss(4, 'aaaa', 0.001, device)

        print("start_epoch: ", start_epoch, accumulate)

        for epoch in range(start_epoch, epochs):

            if experiment_dict['robust'] and epoch == 5:
                model = init_iws(model, experiment_dict, X_test, y_test, device)

            model.train()

            mean_train_loss = 0
            mean_train_acc = 0
            mean_train_f1 = 0
            mean_train_bIoU = 0

            if rank != -1:
                dataloader.sampler.set_epoch(epoch)

            pbar = enumerate(dataloader)
            pbar = tqdm(pbar, total=nb)
        
            optimizer.zero_grad()

            for i, inp in pbar:
                ni = i + nb * epoch
                imgs = inp[0].to(device)
                labels = inp[1].long().to(device)
                labels[labels >= 1] -= 1

                labels_onehot = F.one_hot(labels, num_classes=3).permute((0,3,1,2))

                with amp.autocast(enabled=cuda):
                    outputs = model(imgs)
                    outputs_softmax = [F.softmax(outputs[0][0], dim = 1), F.softmax(outputs[0][1], dim = 1)]
                    losses = loss_funcs.compute_loss(loss_function, outputs_softmax, outputs[0][:2], labels_onehot)

                if experiment_dict['train']['use_CoV'] == True:
                    loss = weightingLoss.forward(losses)
                else:
                    loss = 0.0
                    for j in range(len(losses)):
                        loss += losses[j] * int(experiment_dict['train']['loss_weights'][j])
                loss += 0.6*outputs[-2]

                scaler.scale(loss).backward()

                if ni % accumulate == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    if experiment_dict['train']['ema']==True:
                        ema.update(model)

                acc, f1 = metricsFactory.compute_metrics(outputs_softmax[0], labels, False)
                acc = acc.mean()
                f1 = f1.mean()
                mean_train_loss+=loss.item()
                mean_train_acc+=acc.item()
                mean_train_f1+=f1.item()
     
            if epoch % 1 == 0:
                val_loss, mIoU, mIoU_all, IoU_arr, bIoU_arr, val_acc, val_f1, bIoU, out_images = validate(experiment_dict, testloader, model, loss_function)
                if mIoU >= best_mIoU:
                    best_mIoU = mIoU
                
                createCheckpoint(experiment_dict['savepath'],model,optimizer,epoch,best_mIoU,scheduler)

                if wandb:
                    results_dict={}
                    results_dict['train_loss']=mean_train_loss/len(dataloader)
                    results_dict['train_accuracy']=mean_train_acc/len(dataloader)
                    results_dict['train_f1'] = mean_train_f1/len(dataloader)
                    results_dict['val_loss'] = val_loss
                    results_dict['test_accuracy']=val_acc
                    results_dict['test_f1'] = val_f1
                    results_dict['rail_IoU'] = IoU_arr[1]
                    results_dict['wagon_IoU'] = IoU_arr[2]
                    results_dict['mean_IoU'] = mIoU
                    results_dict['mean_IoU_all'] = mIoU_all
                    results_dict['best_mIoU'] = best_mIoU
                    results_dict['rail_bIoU'] = bIoU_arr[0]
                    results_dict['wagon_bIoU'] = bIoU_arr[1]
                    results_dict['test_bIoU'] = bIoU
                    if epoch % 3 == 0:
                        results_dict['images'] = [wandb.Image(image) for image in out_images]
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
    parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
    parser.add_argument('--validate', action='store_true', help='validate model')
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
    logger.info(opt)

    run_model(opt.hyp,opt,device,wandb)
