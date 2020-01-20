from __future__ import absolute_import

from torch import nn
import torchvision
from models.ClassificationHead import SCH, ECH

__all__ = ['ResNet50']


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        model = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(model.conv1, model.bn1,
                                  model.relu, model.maxpool,
                                  model.layer1, model.layer2,
                                  model.layer3, model.layer4)
        print(self.base)

        self.cls_head = SCH(2048, num_classes)

    def forward(self, x):
        x = self.base(x)

        y = self.cls_head(x)
        return y
