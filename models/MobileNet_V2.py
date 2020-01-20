from __future__ import absolute_import

from torch import nn
import torchvision
from models.ClassificationHead import SCH, ECH

__all__ = ['MobileNet_V2']


class MobileNet_V2(nn.Module):
    def __init__(self, num_classes):
        super(MobileNet_V2, self).__init__()
        model = torchvision.models.mobilenet_v2(pretrained=True)
        self.base = model.features

        self.cls_head = SCH(1280, num_classes)

    def forward(self, x):
        x = self.base(x)

        y = self.cls_head(x)
        return y
