import torch
from torch import nn as nn
import timm
import numpy as np
import math

__all__ = ["NewLoss", "InfoNCELoss", "infonceLoss", "CrossEntropyLoss", "xLoss", "dice_loss", "FocalLoss"]


def compute_score(image_features, text_features):
    image_features /= image_features.norm(dim=1, keepdim=True)
    text_features /= text_features.norm(dim=1, keepdim=True)
    # print("ii: ", image_features.unsqueeze(1).shape)
    # print("text: ", text_features.shape)
    text_probs = (torch.bmm(image_features.unsqueeze(1), text_features) / 0.07).softmax(dim=-1)

    return text_probs


def dice_loss(inputs, targets, num_masks):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks



class CrossEntropyLoss(nn.Module):
    def __init__(self, ):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, predictions, label):
        loss = self.criterion(predictions, label)
        return loss
        
        
class xLoss(nn.Module):
    def __init__(self):
        super(xLoss, self).__init__()
        self.th = 1
        self.th_ = -1
        self.t = 0.99

    def forward(self, img_features, noise_image_features, text_features):

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        noise_image_features = noise_image_features / noise_image_features.norm(dim=-1, keepdim=True)

        N = img_features.shape[0]

        sim = img_features @ text_features.T
        noise_sim = noise_image_features @ text_features.T

        sim_matrix = sim.softmax(dim=-1)
        noise_sim_matrix = noise_sim.softmax(dim=-1)
        
        a = torch.exp(sim[:, 0] / self.t) # normal_score
        b = torch.exp(sim[:, 1] / self.t) # abnormal_score
        c = torch.exp(noise_sim[:, 0] / self.t) # noise_normal_score
        d = torch.exp(noise_sim[:, 1] / self.t) # noise_abnormal_score

        hinge_loss = (self.hinge_loss(sim[:, 0], sim[:, 1]) + self.hinge_loss(noise_sim[:, 0], noise_sim[:, 1])) / 2
        
        con_loss = (self.con_loss(a, b, c) + self.con_loss(d, b, c)) / 2

        crossentropy_loss = (self.crossentropy_loss(a, b) + self.crossentropy_loss(d, c)) / 2

        return con_loss
    
    def hinge_loss(self, pos, neg):
        N = pos.shape[0]
        loss1 = torch.max(self.th - pos, torch.tensor(0)).sum()
        loss2 = torch.max(-self.th_ + neg, torch.tensor(0)).sum()
        return (loss1 + loss2) / (2*N)
    
    def con_loss(self, pos, neg1, neg2):
        N = pos.shape[0]
        loss1 = -torch.log(pos/(pos+neg1)).sum()
        loss2 = -torch.log(pos/(pos+neg2)).sum()
        return (loss1 + loss2) / (2*N)
    
    def crossentropy_loss(self, pos, neg):
        N = pos.shape[0]
        return -torch.log(pos/pos+neg).sum() / N




class infonceLoss(nn.Module):
    def __init__(self, ):
        super(infonceLoss, self).__init__()
        self.th = 1
        self.th_ = -1
        self.t = 0.9


    def forward(self, classtoken, text_features, batch_size):
        img_features = classtoken[0:batch_size]
        noise_image_features = classtoken[batch_size:]
        N = img_features.shape[0]

        sim = img_features @ text_features.T
        noise_sim = noise_image_features @ text_features.T
        
        sim_matrix = sim.softmax(dim=-1)
        noise_sim_matrix = noise_sim.softmax(dim=-1)
        
       # print(sim_matrix, noise_sim_matrix)

        normal_score = torch.exp(sim[:, 0] / self.t)  # 1
        abnormal_score = torch.exp(sim[:, 1] / self.t)  # 2
        noise_normal_score = torch.exp(noise_sim[:, 0] / self.t)  # 3
        noise_abnormal_score = torch.exp(noise_sim[:, 1] / self.t)  # 4
        

        a = normal_score + noise_abnormal_score
        b = abnormal_score + noise_normal_score

        loss = -torch.log(0.5 * a / (a + b)).sum() / N

        loss1 = (torch.log(normal_score / (normal_score + abnormal_score)) + torch.log(
            noise_abnormal_score / (noise_abnormal_score + noise_normal_score))).sum()
        loss2 = (torch.log(normal_score / (normal_score + noise_normal_score)) + torch.log(
            noise_abnormal_score / (noise_abnormal_score + abnormal_score))).sum()

        loss3 = -(torch.log(normal_score / (normal_score + abnormal_score)) + torch.log(
            noise_abnormal_score / (noise_abnormal_score + noise_normal_score))).sum()
        loss4 = -(torch.log(normal_score / (normal_score + noise_normal_score)) + torch.log(
            noise_abnormal_score / (noise_abnormal_score + abnormal_score))).sum()
            
        return (loss3 + loss4) / (2 * N)


class NewLoss(nn.Module):
    def __init__(self):
        super(NewLoss, self).__init__()
        self.th = 1
        self.th_ = -1
        self.t = 0.9

    def forward(self, img_features, noise_image_features, text_features):

        N = img_features.shape[0]

        sim = img_features @ text_features.T
        noise_sim = noise_image_features @ text_features.T

        normal_score = torch.exp(sim[:, 0] / self.t)  # 1
        abnormal_score = torch.exp(sim[:, 1] / self.t)  # 2
        noise_normal_score = torch.exp(noise_sim[:, 0] / self.t)  # 3
        noise_abnormal_score = torch.exp(noise_sim[:, 1] / self.t)  # 4

        a = normal_score + noise_abnormal_score
        b = abnormal_score + noise_normal_score

        loss = -torch.log(0.5 * a / (a + b)).sum() / N

        loss1 = torch.max(self.th - a, torch.tensor(0)).sum() / N
        loss2 = torch.max(-self.th_ + b, torch.tensor(0)).sum() / N

        loss3 = -(torch.log(normal_score / (normal_score + abnormal_score)) + torch.log(
            noise_abnormal_score / (noise_abnormal_score + noise_normal_score))).sum()
        loss4 = -(torch.log(normal_score / (normal_score + noise_normal_score)) + torch.log(
            noise_abnormal_score / (noise_abnormal_score + abnormal_score))).sum()

        return (loss3 + loss4) / (2 * N)


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(InfoNCELoss, self).__init__()
        self.T = temperature

    def forward(self, img_features, noise_idxs, text_feature):
        img_features = nn.functional.normalize(img_features, dim=1)
        text_feature = nn.functional.normalize(text_feature, dim=1)

        noise_features = img_features[noise_idxs]
        noise_features = nn.functional.normalize(noise_features, dim=1)

        sim_noise = noise_features @ text_feature.t()
        sim_noise /= self.T
        sim_noise = torch.exp(sim_noise)
        sim_noise = sim_noise.sum()

        sim = img_features @ text_feature.t()
        sim /= self.T
        sim = torch.exp(sim)
        sim = sim.sum()

        loss = -torch.log(sim_noise / sim)
        return loss
        
        
        
        
        
class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        