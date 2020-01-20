from __future__ import absolute_import

import torch
from torch import nn


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        x = x * (torch.tanh(nn.functional.softplus(x)))
        return x


# Simple Classification Head (SCH)
class SCH(nn.Module):
    def __init__(self, num_chn, num_classes, is_gap=True):
        super(SCH, self).__init__()
        self.isGap = is_gap
        if self.isGap:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.mapping = nn.Sequential(nn.Conv2d(num_chn, 512, 1, bias=False),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU())
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(512, num_classes, bias=False)

    def forward(self, x):
        if self.isGap:
            x = self.gap(x)
        x = self.mapping(x)
        f = x.view(x.size(0), -1)
        f = self.dropout(f)
        y = self.classifier(f)
        return y


# Efficient Classification Head (ECH)
class ECH(nn.Module):
    def __init__(self, num_chn, num_classes, is_gap=True):
        super(ECH, self).__init__()
        self.isGap = is_gap
        if self.isGap:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        self.mapping = nn.Sequential(nn.Conv2d(2*num_chn, 512, 1, bias=False),
                                     nn.BatchNorm2d(512),
                                     Mish())
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(512, num_classes, bias=False)

    def forward(self, x):
        if self.isGap:
            x = torch.cat([self.gmp(x), self.gap(x)], dim=1)
            # x = self.gmp(x) + self.gap(x)
        x = self.mapping(x)
        f = x.view(x.size(0), -1)
        df = self.dropout(f)
        y = self.classifier(df)
        return y
