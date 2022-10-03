# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: Jiang-Jiang Liu
## Email: j04.liu@gmail.com
# Copyright (c) 2020
##
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""SCNet variants"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from models.utils.SCnet_utils import SCBottleneck

__all__ = ['SCNet', 'scnet50', 'scnet101', 'scnet50_v1d', 'scnet101_v1d']

model_urls = {
    'scnet50': 'https://backseason.oss-cn-beijing.aliyuncs.com/scnet/scnet50-dc6a7e87.pth',
    'scnet50_v1d': 'https://backseason.oss-cn-beijing.aliyuncs.com/scnet/scnet50_v1d-4109d1e1.pth',
    'scnet101': 'https://backseason.oss-cn-beijing.aliyuncs.com/scnet/scnet101-44c5b751.pth',
    # 'scnet101_v1d': coming soon...
}


class SCNet(nn.Module):
    """ SCNet Variants Definations
    Parameters
    ----------
    block : Block
        Class for the residual block.
    layers : list of int
        Numbers of layers in each block.
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained SCNet yielding a stride-8 model.
    deep_stem : bool, default False
        Replace 7x7 conv in input stem with 3 3x3 conv.
    avg_down : bool, default False
        Use AvgPool instead of stride conv when
        downsampling in the bottleneck.
    norm_layer : object
        Normalization layer used (default: :class:`torch.nn.BatchNorm2d`).
    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    def __init__(self, block, layers, groups=1, bottleneck_width=32,
                 nOut=1000, dilated=False, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False,
                 avd=False, norm_layer=nn.BatchNorm2d, **kwargs):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        self.input_dim = 1
        # ResNet-D params
        self.inplanes = stem_width*2 if deep_stem else 64
        self.avg_down = avg_down
        self.avd = avd

        self.aug = kwargs['augment']
        self.aug_chain = kwargs['augment_options']['augment_chain']
        # self.inplanes = layers[0]
        self.n_mels = kwargs['n_mels']
        self.kwargs = kwargs

        super(SCNet, self).__init__()
        conv_layer = nn.Conv2d
        if deep_stem:
            self.conv1 = nn.Sequential(
                conv_layer(self.input_dim, stem_width, kernel_size=3,
                           stride=2, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width, kernel_size=3,
                           stride=1, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width*2, kernel_size=3,
                           stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = conv_layer(self.input_dim, 64, kernel_size=7, stride=2, padding=3,
                                    bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, norm_layer=norm_layer)

        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer)
        elif dilation == 2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilation=1, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=2, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, nOut)
        self.instancenorm = nn.InstanceNorm1d(self.n_mels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, norm_layer):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
                                                    ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
                                                    ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, dilation=1, is_first=is_first,
                                norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, dilation=2, is_first=is_first,
                                norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, dilation=dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.no_grad():
            if self.kwargs['features'] == 'melspectrogram':
                x = x + 1e-6
                x = x.log()
                x = x - torch.mean(x, dim=-1, keepdim=True)
            x = self.instancenorm(x).unsqueeze(1)

        assert len(x.size()) == 4  # batch x channel x n_mels x n_frames

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def scnet50(pretrained=False, **kwargs):
    """Constructs a SCNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SCNet(SCBottleneck, [3, 4, 6, 3],
                  deep_stem=False, stem_width=32, avg_down=False,
                  avd=False, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['scnet50']))
    return model


def scnet50_v1d(pretrained=False, **kwargs):
    """Constructs a SCNet-50_v1d model described in
    `Bag of Tricks <https://arxiv.org/pdf/1812.01187.pdf>`_.
    `ResNeSt: Split-Attention Networks <https://arxiv.org/pdf/2004.08955.pdf>`_.

    Compared with default SCNet(SCNetv1b), SCNetv1d replaces the 7x7 conv
    in the input stem with three 3x3 convs. And in the downsampling block,
    a 3x3 avg_pool with stride 2 is added before conv, whose stride is
    changed to 1.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SCNet(SCBottleneck, [3, 4, 6, 3],
                  deep_stem=True, stem_width=32, avg_down=True,
                  avd=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['scnet50_v1d']))
    return model


def scnet101(pretrained=False, **kwargs):
    """Constructs a SCNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SCNet(SCBottleneck, [3, 4, 23, 3],
                  deep_stem=False, stem_width=64, avg_down=False,
                  avd=False, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['scnet101']))
    return model


def scnet101_v1d(pretrained=False, **kwargs):
    """Constructs a SCNet-101_v1d model described in
    `Bag of Tricks <https://arxiv.org/pdf/1812.01187.pdf>`_.
    `ResNeSt: Split-Attention Networks <https://arxiv.org/pdf/2004.08955.pdf>`_.

    Compared with default SCNet(SCNetv1b), SCNetv1d replaces the 7x7 conv
    in the input stem with three 3x3 convs. And in the downsampling block,
    a 3x3 avg_pool with stride 2 is added before conv, whose stride is
    changed to 1.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SCNet(SCBottleneck, [3, 4, 23, 3],
                  deep_stem=True, stem_width=64, avg_down=True,
                  avd=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['scnet101_v1d']))
    return model


def MainModel(nOut=256, **kwargs):
    # Number of filters
    num_filters = [3, 4, 6, 3]
    model = SCNet(SCBottleneck, num_filters, nOut=nOut,
                  deep_stem=False, stem_width=32, avg_down=False,
                  avd=False, dilated=True, **kwargs)
    return model


if __name__ == '__main__':
    images = torch.rand(1, 1, 224, 224).cuda(0)
    model = scnet101(pretrained=True)
    model = model.cuda(0)
    print(model(images).size())
