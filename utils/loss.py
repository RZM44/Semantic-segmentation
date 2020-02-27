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
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        if self.size_average:
            return ce_loss.mean()
        else:
            return ce_loss.sum()

if __name__ == '__main__':
    torch.manual_seed(123)
    inputs = torch.rand(2, 21, 213, 213)
    averageloss = CrossEntropyLoss(size_average=True)
    sumloss = CrossEntropyLoss(size_average=False)
    target = torch.rand(2, 213, 213)
    averageout = averageloss(inputs, target.long())
    sumout = sumloss(inputs, target.long())
    out = F.cross_entropy(inputs, target.long(), reduction='sum')
    print(out.size())
    print(out)
    print(sumout)
    print(averageout)
    

