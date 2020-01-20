from __future__ import absolute_import

from torch import nn
import torchvision
from models.ClassificationHead import SCH, ECH

__all__ = ['SqueezeNet_V2']


class SqueezeNet_V2(nn.Module):
    def __init__(self, num_classes):
        super(SqueezeNet_V2, self).__init__()
        model = torchvision.models.squeezenet1_1(pretrained=True)
        self.base = model.features
        print(self.base)

        self.cls_head = SCH(512, num_classes)

    def forward(self, x):
        x = self.base(x)

        y = self.cls_head(x)
        return y
