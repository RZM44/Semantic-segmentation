import torch.nn as nn
import torch.nn.functional as F
import torch
#Cross Entropy Loss and Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class CrossEntropyLoss(nn.Module):
    def __init__(self, size_average=True, ignore_index=255):
        super(CrossEntropyLoss,self).__init__()
        self.size_average = size_average 
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs,targets, redution='none', ignore_index=self.ignore_index)
        if self.size_average:
            return ce_loss.mean()
        else:
            return ce_loss.sum()
