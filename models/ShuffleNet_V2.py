from __future__ import absolute_import

from torch import nn
import torchvision
from models.ClassificationHead import SCH, ECH

__all__ = ['ShuffleNet_V2']


class ShuffleNet_V2(nn.Module):
    def __init__(self, num_classes):
        super(ShuffleNet_V2, self).__init__()
        shuffle = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        self.base = nn.Sequential(shuffle.conv1,  shuffle.maxpool,
                                  shuffle.stage2, shuffle.stage3,
                                  shuffle.stage4, shuffle.conv5)
        print(self.base)

        self.cls_head = SCH(1024, num_classes)

    def forward(self, x):
        x = self.base(x)

        y = self.cls_head(x)
        return y
