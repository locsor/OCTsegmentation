import torch
from sklearn.metrics import precision_score

import torch.nn.functional as F
import kornia
import cv2
import numpy as np

class Metrics_factory(object):
    def __init__(self):
        self.placeholder = None

    def pixel_acc(self, pred, label):

        pred = F.one_hot(pred, num_classes=3)
        label = F.one_hot(label, num_classes=3)

        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (pred == label).long(), dim=(1,2,3))
        pixel_sum = torch.sum(valid, dim=(1,2,3))

        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)

        return acc

    def f1_score(self, pred, label):

        pred = F.one_hot(pred, num_classes=3)
        label = F.one_hot(label, num_classes=3)

        tp = torch.sum((label * pred), dim=(1,2,3)).to(torch.float32)
        tn = torch.sum(((1 - label) * (1 - pred)), dim=(1,2,3)).to(torch.float32)
        fp = torch.sum(((1 - label) * pred), dim=(1,2,3)).to(torch.float32)
        fn = torch.sum((label * (1 - pred)), dim=(1,2,3)).to(torch.float32)

        epsilon = 1e-7
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        f1 = 2 * (precision*recall) / (precision + recall + epsilon)
        return f1

    def boundary_iou(self, gt, dt, dilation_ratio=0.02):
        def mask_to_boundary_torch(mask, dilation_ratio=0.02):
            mask[mask>0] = 1
            b, h, w = mask.shape
            img_diag = np.sqrt(h ** 2 + w ** 2)
            iterations = int(round(dilation_ratio * img_diag))
            mask = torch.unsqueeze(mask, dim = 1)
            kernel = torch.ones(3, 3).cuda()

            for i in range(iterations):
                dilated_mask = kornia.morphology.erosion(mask, kernel)

            return (mask - dilated_mask)[:,0,...]

        def mask_to_boundary(mask, dilation_ratio=0.02):
            mask = np.uint8(mask.cpu().numpy())
            mask[mask>0] = 1
            b, h, w = mask.shape
            img_diag = np.sqrt(h ** 2 + w ** 2)
            dilation = int(round(dilation_ratio * img_diag))
            if dilation < 1:
                dilation = 1

            new_mask = np.zeros((b,h+2,w+2), dtype=np.uint8)
            out = np.zeros_like(mask)
            for i in range(b):
                new_mask[i] = cv2.copyMakeBorder(mask[i], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
                kernel = np.ones((3, 3), dtype=np.uint8)
                new_mask_erode = cv2.erode(new_mask[i], kernel, iterations=dilation)
                mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
                out[i] = mask[i] - mask_erode

            return out

        boundary_iou = np.zeros(len(gt))
        boundary_iou_array = np.zeros((2, len(gt)))

        gt_onehot = F.one_hot(gt, num_classes=3)
        dt_onehot = F.one_hot(dt, num_classes=3)

        for i in range(1,3):

            gt_boundary = mask_to_boundary(gt_onehot[...,i], dilation_ratio)
            dt_boundary = mask_to_boundary(dt_onehot[...,i], dilation_ratio)
            intersection = np.sum(((gt_boundary * dt_boundary) > 0), axis=(1,2))
            union = np.sum(((gt_boundary + dt_boundary) > 0), axis=(1,2))
            temp = intersection / (union + 1e-6) #np
            boundary_iou += temp
            boundary_iou_array[i-1] = temp
        boundary_iou = boundary_iou / 2
        return boundary_iou, boundary_iou_array

    def compute_metrics(self,pred,targets,calc_bIoU=True):
        _, pred = torch.max(pred, dim=1)

        acc = self.pixel_acc(pred, targets)
        f1 = self.f1_score(pred, targets)
        if calc_bIoU:
            boundary_iou, boundary_iou_array = self.boundary_iou(targets, pred)
            return acc, f1, boundary_iou, boundary_iou_array
        else:
            return acc, f1
