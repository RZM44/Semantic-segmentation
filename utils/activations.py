import torch
import torch.nn as nn
import torch.nn.functional as F

def f_mish(input):
    return input * torch.tanh(F.softplus(input))

class mish(nn.Module):   
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return f_mish(input)

def f_swish(input):
    return input * torch.sigmoid(input)

class swish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):      
        return f_swish(input)
