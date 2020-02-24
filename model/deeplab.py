import torch
import torch.nn as nn
import torch.nn.functional as F

from model.aspp import build_aspp
from model.decoder import build_decoder
from model.resnet import build_resnet

class DeepLab(nn.Module):
    def __init__(self, output_stride=16, num_classes=21):
        super(DeepLab, self).__init__()
        BatchNorm = nn.BatchNorm2d

        self.backbone = build_resnet(output_stride, pretrained=True)
        self.aspp = build_aspp(output_stride)
        self.decoder = build_decoder(num_classes)

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x

    def get_backbone_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_classifier_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

