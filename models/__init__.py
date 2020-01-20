from __future__ import absolute_import

from .SqueezeNet_V2 import *
from .ShuffleNet_V2 import *
from .MobileNet_V2 import *
from .VGGNet16 import *
from .ResNet50 import *
from .DenseNet121 import *

__factory = {
    'squeezenetv2': SqueezeNet_V2,
    'shufflenetv2': ShuffleNet_V2,
    'mobilenetv2': MobileNet_V2,
    'vggnet16': VGGNet16,
    'resnet50': ResNet50,
    'densenet121': DenseNet121,
}


def get_names():
    return __factory.keys()


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
