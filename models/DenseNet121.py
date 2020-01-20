from __future__ import absolute_import

from torch import nn
import torchvision
from models.ClassificationHead import SCH, ECH

__all__ = ['DenseNet121']


class DenseNet121(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet121, self).__init__()
        model = torchvision.models.densenet121(pretrained=True)
        self.base = model.features
        print(self.base)

        self.cls_head = SCH(1024, num_classes)

    def forward(self, x):
        x = self.base(x)

        y = self.cls_head(x)
        return y
