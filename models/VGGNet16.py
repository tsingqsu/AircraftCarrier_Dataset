from __future__ import absolute_import

from torch import nn
import torchvision
from models.ClassificationHead import SCH, ECH

__all__ = ['VGGNet16']


class VGGNet16(nn.Module):
    def __init__(self, num_classes):
        super(VGGNet16, self).__init__()
        model = torchvision.models.vgg16(pretrained=True)
        self.base = model.features

        self.cls_head = SCH(512, num_classes)

    def forward(self, x):
        x = self.base(x)

        y = self.cls_head(x)
        return y
