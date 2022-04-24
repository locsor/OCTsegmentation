import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np
from scipy import ndimage
import cv2

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None, hyp=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.hyp = hyp

    def weight_triangle(img, a=0.5, b=400):
        b, c, h, w, _ = img.shape

        pixel_weights = torch.arange(0, w)
        pixel_weights = torch.tile(pixel_weights, (b, c, h, 1))
        # pixel_weights = np.moveaxis(pixel_weights, 0, 2)

        for i in range(h):
            val = int(np.ceil(i * (ratio - a))) + b
            pixel_weights[h-1-i][(pixel_weights[i] < val/2) + (pixel_weights[i] > w-val/2)] = 0
            
        pixel_weights[pixel_weights > 0] = 1 
        pixel_weights[pixel_weights == 0] = 0.6
        print(pixel_weights.shape)

        return pixel_weights

    def _forward(self, score, targets_onehot):
        cl, ph, pw = score.size(1), score.size(2), score.size(3)
        b, c ,h, w = targets_onehot.size(0), targets_onehot.size(1), targets_onehot.size(2), targets_onehot.size(3)
        #if ph != h or pw != w:
        #    score = F.interpolate(input=score, size=(
        #        h, w), mode='bilinear', align_corners=self.hyp['model']['align_corners'])

        #targets_onehot = F.one_hot(target, num_classes=3).permute(0,3,1,2)
        #probs = F.softmax(score, dim = 1)#.permute(0,2,3,1)
        class_weights = 1 / (torch.sum(targets_onehot, dim=(0,2,3)) + 1e-7)
        class_weights /= torch.sum(class_weights)

        loss = 0
        # pixel_weights = weight_triangle(targets_onehot)
        pixel_weights = torch.ones_like(targets_onehot).float() 
        pixel_weights[:,int(h/3):] *= 0.6

        loss = torch.sum(class_weights * torch.sum(pixel_weights * torch.log(score + 1e-7) * targets_onehot, dim = (0, 2, 3)))
        #for i in range(cl):
        #    loss += class_weights[i] * torch.sum(pixel_weights[:,i,...] * torch.log(score[:,i] + 1e-7) * targets_onehot[:,i])

        return -loss/(b*h*w)

    def forward(self, score, target):

        if self.hyp['model']['num_outputs'] == 1:
            score = [score]

        weights = self.hyp['train']['loss_balance_weights']
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets_onehot, smooth=1):
        if isinstance(inputs, list):
            inputs = inputs[0]

#        targets_onehot = F.one_hot(targets, num_classes=3)
        #probs = F.softmax(inputs, dim = 1).permute(0,2,3,1)
        weights = 1.0 / ((torch.sum(targets_onehot, dim=(0,1,2)) + 1e-7)**2)

        numenator = torch.sum(torch.sum(targets_onehot*inputs, dim = (1,2)) * weights[None, :], dim = 1)
        denominator = torch.sum(torch.sum(targets_onehot+inputs, dim = (1,2)) * weights[None, :], dim = 1)

        loss = 1 - 2.0*(numenator/(denominator+1e-7))

        return torch.mean(loss)

class DistPenDiceLoss(nn.Module):
    def __init__(self):
        super(DistPenDiceLoss, self).__init__()
    def get_dist(self, label):
        res = np.zeros(label.shape)
        posmask = label.astype(np.uint8)
        negmask = 1 - posmask
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                pos_edt = cv2.distanceTransform(posmask[i,j], distanceType = cv2.DIST_L2, maskSize = cv2.DIST_MASK_PRECISE) - 1 #ndimage.distance_transform_edt(posmask[i,j]) - 1
                pos_edt = (pos_edt - np.min(pos_edt)) / (np.max(pos_edt) - np.min(pos_edt) + 1e-7)
                pos_edt *= posmask[i,j]

                neg_edt = cv2.distanceTransform(negmask[i,j], distanceType = cv2.DIST_L2, maskSize = cv2.DIST_MASK_PRECISE) #ndimage.distance_transform_edt(negmask[i,j])
                neg_edt = (neg_edt - np.min(neg_edt)) / (np.max(neg_edt) - np.min(neg_edt) + 1e-7)
                neg_edt *= negmask[i,j]

                temp = neg_edt - pos_edt
                res[i, j] = 1 + temp

        return res

    def forward(self, inputs, targets_onehot):
        if isinstance(inputs, list):
            inputs = inputs[0]

        #targets_onehot = F.one_hot(targets, num_classes=3)

        with torch.no_grad():
            dist = self.get_dist(targets_onehot.cpu().numpy())

        dist = torch.from_numpy(dist).cuda()

        #probs = F.softmax(inputs, dim = 1).permute(0,2,3,1)

        numenator = 2*torch.sum(targets_onehot*inputs, dim = (1,2,3))
        denominator = 2*torch.sum(targets_onehot*inputs, dim = (1,2,3)) + torch.sum(dist*(1-targets_onehot)*inputs, dim = (1,2,3)) \
                      + torch.sum(dist*(1-inputs)*targets_onehot, dim = (1,2,3))

        loss = 1 - (numenator/(denominator+1e-7))

        return torch.mean(loss)


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None, hyp=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.hyp = hyp
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=self.hyp['model']['align_corners'])

        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=self.hyp['model']['align_corners'])
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]

        return pixel_losses.mean()

    def forward(self, score, target):

        if self.hyp['model']['num_outputs'] == 1:
            score = [score]

        weights = self.hyp['train']['loss_balance_weights']
        assert len(weights) == len(score)

        functions = [self._ce_forward] * \
            (len(weights) - 1) + [self._ohem_forward]
        return sum([
            w * func(x, target)
            for (w, x, func) in zip(weights, score, functions)
        ]).mean()

def create_loss_function(hyp,class_weights=None):
    loss_type = hyp['train']['loss_type']
    loss_type_mt = hyp['train']['loss_type_mt']

    if hyp['model']['model_type'] == 'Segmentation':
        if loss_type=='CE':
            loss=CrossEntropy(hyp['train']['ignore_label'], class_weights)
        if loss_type=='Ohem':
            loss=OhemCrossEntropy(hyp['train']['ignore_label'], hyp['train']['loss_ohemthres'],
                                  hyp['train']['loss_ohemkeep'], class_weights, hyp)
        if loss_type=='Dice':
            loss = DiceLoss()
        if loss_type=='Boundary':
            loss = DistPenDiceLoss()
    else:
        loss=[]
        for i in range(len(loss_type_mt)):
            if loss_type_mt[i] == 'CE':
                loss_s = CrossEntropy(hyp['train']['ignore_label'], class_weights, hyp)
                loss.append(loss_s)
            if loss_type_mt[i] == 'Ohem':
                loss_s = OhemCrossEntropy(hyp['train']['ignore_label'], hyp['train']['loss_ohemthres'],
                                          hyp['train']['loss_ohemkeep'], class_weights, hyp)
                loss.append(loss_s)
            if loss_type_mt[i] == 'Dice':
                loss_s = DiceLoss()
                loss.append(loss_s)
            if loss_type_mt[i] == 'Boundary':
                loss_s = DistPenDiceLoss()
                loss.append(loss_s)
    return loss

def compute_loss(loss_function, pred, targets):
    device = targets.device
    if type(loss_function)!=list:
        loss_function=loss_function.to(device)
        loss = loss_function(pred,targets)
    else:
        loss = []
        for loss_f in loss_function:
            #loss_f=loss_f.to(device)
            loss.append(torch.unsqueeze(loss_f(pred, targets),0)) 

    return loss
