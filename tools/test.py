import cv2
import torch
import yaml
import pathlib
import _init_paths as _init_paths
import ruamel.yaml
import argparse
import os, ast
import operator
import sys, time
import logging, random, csv
import numpy as np
import torch.nn.functional as F
from utils.utils import get_confusion_matrix
from utils.utils import select_device, init_seeds
import utils.metrics as metrics
import kindle
from kindle import Model, TorchTrainer
import albumentations as A
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2

def resume(model, path):
    pretrained_state = torch.load(path, map_location='cpu')
    optimizer = pretrained_state['optimizer_state_dict']
    scheduler = pretrained_state['scheduler']
    start_epoch = int(pretrained_state['last_epoch'])

    model_dict = model.state_dict()
    model_dict.update(pretrained_state['model_state_dict'])
    model.load_state_dict(model_dict, strict = False)

    return model, optimizer, scheduler, start_epoch

def get_x_y(folder):
    frames = []
    masks = []

    for _, dirs, _ in os.walk(folder):
        for directory in dirs:
            frames_dir = []
            frames_dir_num = []
            for _, _, file in os.walk(folder + '/' + directory):
                for name in file:
                    if name.endswith('_frame.png'):
                        frames_dir += [folder + '/' + directory + '/' + name]
                        frames_dir_num += [int(name[:-10])]
                        
            frames_dir_num, frames_dir = zip(*sorted(zip(frames_dir_num, frames_dir)))
            frames += list(frames_dir)
            
    masks = [file.replace('frame', 'mask') for file in frames]
    return frames, masks

def transform(sample):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    base_size = (640, 960)

    crop_x1 = random.randint(0,300)
    crop_x2 = random.randint(0,300)
    crop_y1 = random.randint(0,200)
    crop_y2 = random.randint(0,200)
    transform = A.Compose([A.Resize(height=base_size[0],width=base_size[1]),
                           A.Normalize(mean, std),
                           ToTensorV2()])

    sample = transform(image=sample['image'], mask=sample['mask'])
    return sample

def inference(model, device, folder, output):
    model.eval()
    img_size = experiment_dict['dataset']['img_size_test']

    bIoU_arr = []
    acc_arr = []
    confusion_matrix = np.zeros((3, 3, 1))

    with torch.no_grad():
        imgs, labels = get_x_y(folder)

        for i in tqdm(range(len(imgs))):
            img_name = imgs[i]
            lable_name = labels[i]

            img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            img_reg = img.copy()
            img_reg = cv2.resize(img_reg, (960,640))
            img[img == 0] = 255
            label = cv2.imread(lable_name)[:,:,0]

            sample={"image":img,"mask":label}
            sample = transform(sample)
            size = sample['mask'].size()

            result = model(torch.unsqueeze(sample['image'], dim=0).to(device))

            result = torch.softmax(result[0][0], dim=1)
            _, result = torch.max(result, dim=1)

            out = np.uint8(F.one_hot(result, num_classes=3).cpu().numpy()[0] * 255)
            out[...,0] = 0

            out = cv2.addWeighted(img_reg,0.7,out,0.3,0)

            # folder = img_name.split('/')[3]
            # if os.path.isdir(output + folder) is False:
                # os.mkdir(output + folder)
            # img_name = img_name.replace(folder, output)
            img_name = output + str(i) + '.png'
            cv2.imwrite(img_name, out)

def call_inference(experiment_dict, model, folder, output, device):
    model,_,_,_ = resume(model, experiment_dict['resumepath'])
    inference(model, device, folder, output)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str, default='./lib/config/train_config.yaml',
                        help='hyperparameters path')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--folder', help='folder with image')
    parser.add_argument('--output', help='folder to write image into')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    opt = parser.parse_args()
    device = select_device(opt.device)
    folder = opt.folder
    output = opt.output
    rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.hyp) as f:
        experiment_dict = yaml.load(f, Loader=yaml.FullLoader)

    yaml_ruamel = ruamel.yaml.YAML()
    with open("./tools/kindle/ddrnet_23.yaml") as fp:
        data = yaml_ruamel.load(fp)

    with open("./tools/kindle/temp.yaml", 'w') as yaml_file:
        yaml_ruamel.dump(data, yaml_file)

    model = Model("./tools/kindle/temp.yaml",verbose=False).to(device) #init model
    model,_,_,_ = resume(model, experiment_dict['resumepath'])

    metrics = metrics.Metrics_factory()

    call_inference(experiment_dict, model, folder, output, device)
