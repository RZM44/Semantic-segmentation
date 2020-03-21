import torch
import torch.nn as nn
import torch.nn.functional as F

from model.aspp import build_aspp
from model.decoder import build_decoder
from model.resnet import build_resnet
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class DeepLab(nn.Module):
    def __init__(self, output_stride=16, num_classes=21, sy_bn=True):
        super(DeepLab, self).__init__()
        if(sy_bn == True):
          BatchNorm = SynchronizedBatchNorm2d
        else:
          BatchNorm = nn.BatchNorm2d

        self.backbone = build_resnet(output_stride, BatchNorm, pretrained=True)
        self.aspp = build_aspp(output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, BatchNorm)

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
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_classifier_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
if __name__ == "__main__":
    #from aspp import build_aspp
    #from decoder import build_decoder
    #from resnet import build_resnet
    #from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
    model = DeepLab(output_stride=16, sy_bn=False)
    model.eval()
    input = torch.rand(1, 3, 213, 213)
    output = model(input)
    print(output.size())
