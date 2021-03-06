import cv2
import numpy as np
import os, random
from PIL import Image

import torch
from torch.nn import functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as torch_tr

from utils.utils import torch_distributed_zero_first
from scipy import ndimage
from skimage.filters import gaussian

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_sampler(dataset):
    from utils.distributed import is_distributed
    if is_distributed():
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset)
    else:
        return None

def get_dataset(experiment_dict, X, y, batch_size, world_size=1, workers=1, rank=1, val=False, set_var=False, cuda=True):
    data_dict = experiment_dict.copy()
    crop_size = (experiment_dict['dataset']['img_size'][0], experiment_dict['dataset']['img_size'][1])
    base_size = data_dict['dataset']['base_size']

    if val:
        data_dict['train']['shuffle'] = False
        data_dict['dataset']['train_num_samples'] = data_dict['dataset']['test_num_samples']
        base_size = data_dict['dataset']['base_size']

    dataset = Dataset(X, y,
                      num_samples = data_dict['dataset']['train_num_samples'],
                      num_classes = data_dict['model']['num_classes'],
                      ignore_label = data_dict['train']['ignore_label'],
                      base_size = base_size,
                      crop_size = crop_size,
                      mean = [data_dict['dataset']['mean0'],
                              data_dict['dataset']['mean1'],
                              data_dict['dataset']['mean2']],
                      std =  [data_dict['dataset']['std0'],
                              data_dict['dataset']['std1'],
                              data_dict['dataset']['std2']],
                      val = val,
                      set_var = set_var)

    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = get_sampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=data_dict['train']['shuffle'] and sampler is None,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
        sampler=sampler)

    return dataset, dataloader

class Dataset():
    def __init__(self,
                 filenames_img,
                 filenames_mask,
                 num_samples=None,
                 num_classes=4,
                 ignore_label=255,
                 base_size=[2048,2048],
                 crop_size=(512, 1024),
                 downsample_rate=1,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 test=False,
                 val=False,
                 set_var=False):

        self.filenames_img = filenames_img
        self.filenames_mask = filenames_mask
        self.num_classes = num_classes
        self.test = test
        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.downsample_rate = downsample_rate
        self.mean = mean
        self.std = std
        self.val = val
        self.set_var = set_var

        if num_samples:
            self.filenames_img = self.filenames_img[:num_samples]
            self.filenames_mask = self.filenames_mask[:num_samples]

    def __len__(self):
        return len(self.filenames_img)

    def transform(self, sample, train):
        transform = A.Compose([A.Normalize(self.mean, self.std),
                               A.Resize(height=self.crop_size[0],width=self.crop_size[1]),
                               ToTensorV2()])

        if train:

            transform = A.Compose([A.HorizontalFlip(p=0.5),
                                   A.RandomGamma(gamma_limit=(32, 96), p=0.5),
                                   A.RandomBrightnessContrast(p=0.5),
                                   A.HueSaturationValue(p=0.5),
                                   A.OneOf([A.MotionBlur(blur_limit=25, p=1),
                                            A.Blur(blur_limit=25, p=1)], p=0.2),
                                   A.RandomScale(scale_limit=(-0.5, -0.2), p=0.5, interpolation=1),
                                   A.PadIfNeeded(self.base_size[0], self.base_size[1], border_mode=cv2.BORDER_CONSTANT),
                                   A.Resize(height=self.crop_size[0],width=self.crop_size[1]),
                                   A.Normalize(self.mean, self.std),
                                   ToTensorV2()])

        sample = transform(image=sample['image'], mask=sample['mask'])
        return sample

    def __getitem__(self, index):
        item = self.filenames_img[index]
        image = cv2.imread(item, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(self.filenames_mask[index], 0)
        if np.max(np.unique(label)) in [4,5]:
            label[label==5] = 4 #main rail
            label[label==2] = 3 #wagon directly in front
            label[label==4] = 2 #periphery rail
        sample={"image":image.copy(),"mask":label}

        if self.test or (self.val and not self.set_var):
            # image = cv2.resize(image, (960, 640))
            sample=self.transform(sample,False)

            return sample['image'], sample['mask'], item#image[:,:,::-1].copy()

        elif self.set_var:
            image_orig = image.copy()
            image = np.uint8(self.photometric_transform(image))

            sample={"image":image,"mask":label}
            sample=self.transform(sample,False)
            sample_orig={"image":image_orig,"mask":label}
            sample_orig=self.transform(sample_orig,False)

            return [sample_orig['image'], sample['image']], np.array(self.base_size), item

        sample = self.transform(sample,True)

        return sample['image'], sample['mask'], np.array(self.crop_size), item

    def label_transform(self, label):
        return np.array(label).astype('int32')

    def photometric_transform(self, img):
        brightness = 0.8
        contrast = 0.8
        saturation = 0.8
        hue = 0.3

        img = Image.fromarray(np.uint8(img))

        brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
        contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
        saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
        hue_factor = np.random.uniform(-hue, hue)

        transforms = []
        transforms.append(torch_tr.Lambda(lambda img: TF.adjust_brightness(img, brightness_factor)))
        transforms.append(torch_tr.Lambda(lambda img: TF.adjust_contrast(img, contrast_factor)))
        transforms.append(torch_tr.Lambda(lambda img: TF.adjust_saturation(img, saturation_factor)))
        transforms.append(torch_tr.Lambda(lambda img: TF.adjust_hue(img, hue_factor)))

        np.random.shuffle(transforms)
        transform = torch_tr.Compose(transforms)

        img = transform(img)

        sigma = 0.15 + random.random() * 1.15
        blurred_img = gaussian(np.array(img), sigma=sigma, multichannel=True)
        blurred_img *= 255

        return np.asarray(blurred_img)
