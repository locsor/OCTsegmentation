import cv2
import torch
import yaml
import pathlib
import _init_paths as _init_paths
import ruamel.yaml
import argparse
import os
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

# def get_x_y(filename):
#     img_files = []
#     mask_files = []

#     filenames = []
#     with open(filename, newline='') as f:
#         reader = csv.reader(f)
#         filenames = list(reader)
#         filenames = [item for sublist in filenames for item in sublist]
#         file_nums = []
#         print(filenames[:100])
#         for i in range(len(filenames)):
#             if filenames[i][-13:-10].isdigit():
#                 file_nums.append(int(filenames[i][-13:-10]))
#             elif filenames[i][-12:-10].isdigit():
#                 file_nums.append(int(filenames[i][-12:-10]))
#             elif filenames[i][-11:-10].isdigit():
#                 file_nums.append(int(filenames[i][-11:-10]))

#         file_idx = list(range(len(file_nums)))
#         file_nums, file_idx = zip(*sorted(zip(file_nums, file_idx)))

#     for idx in file_idx:
#         file = filenames[idx]#[0]
#         img_files.append(file)
#         mask_name = file.replace('/img/', '/masks/')
#         # mask_name = file.replace('_frame', '_mask')
#         mask_files.append(mask_name)

#     return np.asarray(img_files), np.asarray(mask_files)

def get_x_y(filename):
    img_files = []
    mask_files = []

    filenames = []
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        filenames = list(reader)

    for i in range(len(filenames)):
        file = filenames[i][0]
        img_files.append('./data/' + file)
        mask_name = file.replace('/img/', '/masks/')
        mask_files.append('./data/' + mask_name)

    return np.asarray(img_files), np.asarray(mask_files)

def transform(sample):
    mean = [0.473, 0.482, 0.469]
    std = [0.174, 0.177, 0.175]
    base_size = (640, 960)

    transform = A.Compose([A.Resize(height=base_size[0],width=base_size[1]),
                           A.Normalize(mean, std),
                           ToTensorV2()])

    sample = transform(image=sample['image'], mask=sample['mask'])
    return sample

def inference(model, device):
    model.eval()
    img_size = (960, 640)
    # folder = [x[0] for x in os.walk('./data/logs/') if 'out' not in x[0] and '_' in x[0]]
    folder = ['./data/test']

    bIoU_arr = []
    acc_arr = []
    confusion_matrix = np.zeros((3, 3, 1))

    with torch.no_grad():
        for f in folder:
            print(f)
            imgs, labels = get_x_y(f+'.csv')
            imgs = imgs[:256]
            lables = labels[:256]

            for i in tqdm(range(len(imgs))):
                # imgs[i] = '.' + imgs[i][1:]
                # labels[i] = '.' + labels[i][1:]
                img_name = imgs[i][2:]
                lable_name = labels[i][2:]

                img = cv2.imread(img_name, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                label = cv2.imread(lable_name, 0)
                label[label >= 1] -= 1
                # img_gray = cv2.imread(img_name, 0)

                # img_gray[img_gray>0] = 1
                # img_shape = img_gray.shape
                # y1 = list(img_gray[256,:]).index(1)
                # y2 = img_shape[1] - list(img_gray[256,:])[::-1].index(1)
                # x1 = list(img_gray[:,400]).index(1)
                # x2 = img_shape[0] - list(img_gray[:,400])[::-1].index(1)
                # if x1 != 0:
                #     continue
                    # img = img[x1:x2, y1:y2]
                    # img = img[:,:,::-1]
                    # label = label[x1:x2, y1:y2]

                # img = cv2.resize(img, img_size)
                # label = cv2.resize(label, img_size)
                # img = center_crop(img, img_size)
                # label = center_crop(label, img_size)
                # size = label.shape

                # img_reg = img.copy()
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img = input_transform(img).to(device)

                sample={"image":img,"mask":label}
                sample = transform(sample)
                size = sample['mask'].size()

                result = model(torch.unsqueeze(sample['image'], dim=0).to(device))

                result = torch.softmax(result[0][0], dim=1)
                # label = torch.unsqueeze(torch.from_numpy(label), dim = 0)

                # confusion_matrix = get_confusion_matrix(label,
                #                     result,
                #                     size,
                #                     3,
                #                     255)

                # pos = confusion_matrix.sum(1)
                # res = confusion_matrix.sum(0)
                # tp = np.diag(confusion_matrix)
                # IoU_array = (tp / np.maximum(1.0, pos + res - tp))

#                print(metrics.boundary_iou(torch.unsqueeze(sample['mask'], dim=0), result))
                _, result = torch.max(result, dim=1)

                out = np.uint8(F.one_hot(result, num_classes=3).cpu().numpy()[0] * 255)
                #label = np.uint8(np.eye(3)[label[0]]) * 127
                out[...,0] = 0
                #label[...,0] = 0

                # result = cv2.resize(result, img.shape[:2][::-1], interpolation = cv2.INTER_NEAREST)
                out = cv2.addWeighted(img,0.7,out,0.3,0)
                # out = cv2.addWeighted(out,0.7,label,0.3,0)

                cv2.imwrite('./output/test/' + str(i) + '.png', out)
                # cv2.imwrite(f + '_out/' + str(i) + '.png', out)

def call_inference(experiment_dict, model, device):
    model,_,_,_ = resume(model, experiment_dict['resumepath'])
    inference(model, device)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str, default='./lib/config/train_config.yaml',
                        help='hyperparameters path')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    opt = parser.parse_args()
    device = select_device(opt.device)
    rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.hyp) as f:
        experiment_dict = yaml.load(f, Loader=yaml.FullLoader)

    yaml_ruamel = ruamel.yaml.YAML()
    with open("./tools/kindle/ddrnet_23_slim.yaml") as fp:
        data = yaml_ruamel.load(fp)

    with open("./tools/kindle/temp.yaml", 'w') as yaml_file:
        yaml_ruamel.dump(data, yaml_file)

    model = Model("./tools/kindle/temp.yaml",verbose=False).to(device) #init model
    model,_,_,_ = resume(model, experiment_dict['resumepath'])

    metrics = metrics.Metrics_factory()

    call_inference(experiment_dict, model, device)
