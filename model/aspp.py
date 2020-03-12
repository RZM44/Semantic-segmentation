import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.activations.py import *
class ASPP_Module(nn.Module):
    def __init__(self, inplanes, planes, dilation,activation = 'relu'):
        super(ASPP_Module, self).__init__()
        self.activation = activation
        
        if self.activation == 'relu':
            self.act = nn.ReLU()
            
        if self.activation == 'swish':
            self.act = swish()
            
        if self.activation == 'mish':
            self.act = mish()
        if dilation == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = dilation
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
       

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.act(x)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, output_stride, inplanes=2048,activation = 'relu'):
        super(ASPP, self).__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU()

        if self.activation == 'swish':
            self.act = swish()

        if self.activation == 'mish':
            self.act = mish()
        self.aspp1 = ASPP_Module(inplanes, 256, dilation=dilations[0])
        self.aspp2 = ASPP_Module(inplanes, 256, dilation=dilations[1])
        self.aspp3 = ASPP_Module(inplanes, 256, dilation=dilations[2])
        self.aspp4 = ASPP_Module(inplanes, 256, dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        
        #self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        # below structure not show in orginal paper
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        # x = self.dropout(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(output_stride):
    return ASPP(output_stride)
