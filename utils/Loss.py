import torch.nn as nn
import torch.nn.functional as F
import torch
#Cross Entropy Loss and Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, size_average=True, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss

class CrossEntropyLoss(nn.Module):
    def __init__(self,size_average=True,ignore_index=255):
        super(CrossEntropyLoss,self).__init__()
        self.size_average = size_average 
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs,targets,size_average=True,ignore_index=self.ignore_index)
        return ce_loss
