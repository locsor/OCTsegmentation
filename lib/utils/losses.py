import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from scipy import ndimage
import cv2

class CrossEntropy(nn.Module):
    def __init__(self, hyp, device):
        super(CrossEntropy, self).__init__()
        self.hyp = hyp
        self.device = device

    def weight_triangle(self, img, a=0.5, offset=0):
        b, c, h, w = img.shape

        pixel_weights = torch.arange(0,w).to(self.device)#cuda()
        pixel_weights = torch.tile(pixel_weights, (h,1))
        ratio = w/h

        for i in range(h):
            val = int(np.ceil(i*(ratio - a))) + offset
            pixel_weights[i][(pixel_weights[i] < val/2) + (pixel_weights[i] > w-val/2)] = 0

        pixel_weights[pixel_weights > 0] = 1
        step = 0.4/h
        temp = torch.arange(1,0.6,-step).to(self.device)#.cuda()#.view(1,h)
        pixel_weights = pixel_weights * temp[:, None]
        pixel_weights[pixel_weights == 0] = 0.5
        pixel_weights = torch.tile(pixel_weights, (b, c, 1, 1))

        return pixel_weights

    def _forward(self, score, targets_onehot):
        cl, ph, pw = score.size(1), score.size(2), score.size(3)
        b, c ,h, w = targets_onehot.size(0), targets_onehot.size(1), targets_onehot.size(2), targets_onehot.size(3)

        class_weights = 1 / (torch.sum(targets_onehot, dim=(0,2,3)) + 1e-7)
        class_weights /= torch.sum(class_weights)

        loss = torch.sum(class_weights[None,:,None,None] * torch.log(score + 1e-7) * targets_onehot, dim = 1)
        
        return -torch.mean(loss)

    def forward(self, pred, score, target):

        if self.hyp['model']['num_outputs'] == 1:
            pred = [pred]

        weights = self.hyp['train']['loss_balance_weights']
        assert len(weights) == len(pred)

        # return [self._forward(x, target) for x in pred]
        return sum([w * self._forward(x, target) for (w, x) in zip(weights, pred)])

class DiceLoss(nn.Module):
    def __init__(self, hyp=None):
        super(DiceLoss, self).__init__()
        self.hyp = hyp

    def _forward(self, inputs, targets_onehot, smooth=1):
        if isinstance(inputs, list):
            inputs = inputs[0]

        weights = 1.0 / ((torch.sum(targets_onehot, dim=(0,1,2)) + 1e-7)**2)

        numenator = torch.sum(torch.sum(targets_onehot*inputs, dim = (1,2)) * weights[None, :], dim = 1)
        denominator = torch.sum(torch.sum(targets_onehot+inputs, dim = (1,2)) * weights[None, :], dim = 1)

        loss = 1 - 2.0*(numenator/(denominator+1e-7))

        return torch.mean(loss)

    def forward(self, score, target):

        if self.hyp['model']['num_outputs'] == 1:
            score = [score]

        weights = self.hyp['train']['loss_balance_weights']
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])

class TverskyLoss(nn.Module):
    def __init__(self, hyp=None):
        super(TverskyLoss, self).__init__()
        self.hyp = hyp

    def _forward(self, inputs, targets_onehot, smooth=1):
        if isinstance(inputs, list):
            inputs = inputs[0]

        alpha = 0.1
        beta = 0.9
        weights = 1.0 / ((torch.sum(targets_onehot, dim=(0,1,2)) + 1e-7)**2)

        inputs = F.sigmoid(inputs) * weights[None, :]

        TP = (inputs * targets_onehot).sum()    
        FP = ((1-targets_onehot) * inputs).sum()
        FN = (targets_onehot * (1-inputs)).sum()

        Tversky = (TP + 1e-7) / (TP + alpha*FP + beta*FN + 1e-7)  
        
        return 1 - Tversky

    def forward(self, pred, score, target):

        if self.hyp['model']['num_outputs'] == 1:
            score = [score]

        weights = self.hyp['train']['loss_balance_weights']
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])

class LovaszLoss(nn.Module):
    def __init__(self, hyp=None):
        super(LovaszLoss, self).__init__()
        self.hyp = hyp

    def lovasz_grad(self, gt_sorted):
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def lovasz_softmax_flat(self, probas, labels, classes='present'):
        C = probas.size(1)
        losses = []
        class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
        for c in class_to_sum:
            fg = (labels == c).float() # foreground for class c
            if (classes is 'present' and fg.sum() == 0):
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError('Sigmoid output possible only with 1 class')
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (Variable(fg) - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, Variable(self.lovasz_grad(fg_sorted))))

        return torch.mean(torch.stack(losses))

    def lovasz_softmax(self, probas, labels, classes='all', ignore=None):
        loss = self.lovasz_softmax_flat(*self.flatten_probas(probas, labels, ignore), classes=classes)
        return loss

    def flatten_probas(self, probas, labels, ignore=None):
        if probas.dim() == 3:
            # assumes output of a sigmoid layer
            B, H, W = probas.size()
            probas = probas.view(B, 1, H, W)
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        labels = labels.view(-1)
        if ignore is None:
            return probas, labels
        valid = (labels != ignore)
        vprobas = probas[valid.nonzero().squeeze()]
        vlabels = labels[valid]
        return vprobas, vlabels

    def forward(self, pred, score, target):

        if self.hyp['model']['num_outputs'] == 1:
            score = [score]

        weights = self.hyp['train']['loss_balance_weights']
        assert len(weights) == len(score)

        target = torch.argmax(target, dim=1)

        return sum([w * self.lovasz_softmax(x, target) for (w, x) in zip(weights, pred)])

class DistPenDiceLoss(nn.Module):
    def __init__(self, hyp, device):
        super(DistPenDiceLoss, self).__init__()
        self.hyp = hyp
        self.device = device
    def get_dist(self, label):
        posmask = label.astype(np.uint8)
        negmask = 1 - posmask
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                pos_edt = cv2.distanceTransform(posmask[i,j], distanceType = cv2.DIST_L2, maskSize = 0)  #ndimage.distance_transform_edt(posmask[i,j]) - 1
                pos_edt = 1 - (pos_edt - np.min(pos_edt)) / (np.max(pos_edt) - np.min(pos_edt) + 1e-7)
                pos_edt *= posmask[i,j]

                neg_edt = cv2.distanceTransform(negmask[i,j], distanceType = cv2.DIST_L2, maskSize = 0) #ndimage.distance_transform_edt(negmask[i,j])
                neg_edt = 1 - (neg_edt - np.min(neg_edt)) / (np.max(neg_edt) - np.min(neg_edt) + 1e-7)
                neg_edt *= negmask[i,j]

        res = neg_edt + pos_edt

        return res

    def _forward(self, inputs, targets_onehot):

        with torch.no_grad():
            dist = self.get_dist(targets_onehot.cpu().numpy())

        dist = torch.from_numpy(dist).to(self.device)#.cuda()

        numenator = 2*torch.sum(targets_onehot*inputs, dim = (1,2,3))
        denominator = 2*torch.sum(targets_onehot*inputs, dim = (1,2,3)) + torch.sum(dist*(1-targets_onehot)*inputs, dim = (1,2,3)) \
                      + torch.sum(dist*(1-inputs)*targets_onehot, dim = (1,2,3))

        loss = 1 - (numenator/(denominator+1e-7))

        return torch.mean(loss)

    def forward(self, pred, score, target):

        if self.hyp['model']['num_outputs'] == 1:
            pred = [pred]

        weights = self.hyp['train']['loss_balance_weights']
        assert len(weights) == len(pred)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, pred)])


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

    def _ce_forward(self, pred, targets, reduce=True):

        class_weights = 1 / (torch.sum(targets, dim=(0,2,3)) + 1e-7)
        class_weights /= torch.sum(class_weights)

        loss = class_weights[None,:,None,None] * torch.log(pred + 1e-7) * targets
        
        if reduce:
            loss = torch.sum(loss, dim = 1)
            return -torch.mean(loss)
        else:
            return -loss

    def _ohem_forward(self, pred, target, **kwargs):
        ph, pw = pred.size(2), pred.size(3)
        h, w = target.size(1), target.size(2)

        pixel_losses = self._ce_forward(pred, target, False).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]

        return pixel_losses.mean()

    def forward(self, pred, score, target):

        if self.hyp['model']['num_outputs'] == 1:
            pred = [pred]

        weights = self.hyp['train']['loss_balance_weights']

        assert len(weights) == len(pred)

        functions = [self._ohem_forward] * \
            (len(weights) - 1) + [self._ce_forward]

        loss = sum([
            w * func(x, target)
            for (w, x, func) in zip(weights, pred, functions)
        ]).mean()
        return loss

class CoVWeightingLoss(nn.Module):
    def __init__(self, num_losses, mean_sort, mean_decay_param, device):
        super(CoVWeightingLoss, self).__init__()

        self.mean_decay = True if mean_sort == 'decay' else False
        self.mean_decay_param = mean_decay_param
        self.num_losses = num_losses

        self.current_iter = -1
        self.alphas = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(device)

        self.running_mean_L = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(device)
        self.running_mean_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(device)
        self.running_S_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(device)
        self.running_std_l = None
        self.device = device

    def forward(self, losses):
        L = torch.tensor(losses, requires_grad=False).to(self.device)

        self.current_iter += 1
        L0 = L.clone() if self.current_iter == 0 else self.running_mean_L
        l = L / L0

        if self.current_iter <= 1:
            self.alphas = torch.ones((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(
                self.device) / self.num_losses
        else:
            ls = self.running_std_l / self.running_mean_l
            self.alphas = ls / torch.sum(ls)

        if self.current_iter == 0:
            mean_param = 0.0
        elif self.current_iter > 0 and self.mean_decay:
            mean_param = self.mean_decay_param
        else:
            mean_param = (1. - 1 / (self.current_iter + 1))

        x_l = l.clone().detach()
        new_mean_l = mean_param * self.running_mean_l + (1 - mean_param) * x_l
        self.running_S_l += (x_l - self.running_mean_l) * (x_l - new_mean_l)
        self.running_mean_l = new_mean_l

        running_variance_l = self.running_S_l / (self.current_iter + 1)
        self.running_std_l = torch.sqrt(running_variance_l + 1e-8)

        x_L = L.clone().detach()
        self.running_mean_L = mean_param * self.running_mean_L + (1 - mean_param) * x_L

        self.alphas *= 5
        weighted_losses = [self.alphas[i] * losses[i] for i in range(len(losses))]
        loss = sum(weighted_losses)
        return loss

def create_loss_function(hyp, device, class_weights=None):
    loss_type_mt = hyp['train']['loss_type_mt']

    if hyp['model']['model_type'] == 'Single':
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
                loss_s = CrossEntropy(hyp, device)
                loss.append(loss_s)
            if loss_type_mt[i] == 'Ohem':
                loss_s = OhemCrossEntropy(hyp['train']['ignore_label'], hyp['train']['loss_ohemthres'],
                                          hyp['train']['loss_ohemkeep'], class_weights, hyp)
                loss.append(loss_s)
            if loss_type_mt[i] == 'Dice':
                loss_s = DiceLoss(hyp)
                loss.append(loss_s)
            if loss_type_mt[i] == 'Tversky':
                loss_s = TverskyLoss(hyp)
                loss.append(loss_s)
            if loss_type_mt[i] == 'Lovasz':
                loss_s = LovaszLoss(hyp)
                loss.append(loss_s)
            if loss_type_mt[i] == 'Boundary':
                loss_s = DistPenDiceLoss(hyp, device)
                loss.append(loss_s)
    return loss

def compute_loss(loss_function, pred, score, targets):
    device = targets.device
    if type(loss_function)!=list:
        loss_function=loss_function.to(device)
        loss = loss_function(pred,targets)
    else:
        loss = []
        for loss_f in loss_function:
            loss.append(torch.unsqueeze(loss_f(pred, score, targets),0)) 

    return loss
