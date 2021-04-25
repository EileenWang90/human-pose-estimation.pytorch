# ------------------------------------------------------------------------------
# Reference https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/models/pose_resnet.py
# Written by Yiting Wang  2021.04.21
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict

import re

from typing import Callable, Any, List
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

# from https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py
import math

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=True),  #bias由原来的False改为True
        # nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=True),
        # nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def make_divisible(x, divisible_by=8):
    import numpy as np
    #print(x, np.ceil(x * 1. / divisible_by))
    #print(int(np.ceil(x * 1. / divisible_by) * divisible_by))
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=True),
                # nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=True),
                # nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=True),
                # nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=True),
                # nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=True),
                # nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!  ?
        self.last_channel = make_divisible(last_channel * width_mult) 
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) 
            #print("s=",s,"    output_channel=",output_channel)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet_v2(pretrained=True):
    model = MobileNetV2(width_mult=0.5)

    if pretrained:
        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url
        state_dict = load_state_dict_from_url(
            'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', progress=True)
        model.load_state_dict(state_dict)
    return model


interverted_residual_setting = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]

class PoseMobileNet(nn.Module):

    def __init__(self, block, layers, cfg, 
        stages_repeats: List[int],
        stages_out_channels: List[int],
        num_classes: int = 1000,
        inverted_residual: Callable[..., nn.Module] = InvertedResidual,
        **kwargs ) -> None:

        self.inplanes = 128
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        super(PoseMobileNet, self).__init__()

        block = InvertedResidual
        input_channel = 32
        width_mult = 0.5

        input_channel = make_divisible((input_channel * width_mult)) # first channel is always 32!  ?
        self.features = [conv_bn(3, input_channel, 2)]
        #self.conv1 = conv_bn(3, input_channel, 2)
        #self.features = []
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible((c * width_mult))
            #print("s=",s,"    output_channel=",output_channel)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # in order to make the channel consistent  channel{v2_1.0: 464 -> resnet self.inplanes: 128}
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channel, self.inplanes, kernel_size=1, stride=1, padding=0, bias=True), 
            # nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
        )
     
        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0,
            bias=False
        )

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):  # Idea comes from Mobilenet(depthwise separable convolution)
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(  # depthwise_deconv
                    in_channels=self.inplanes,
                    out_channels=self.inplanes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    groups=128,
                    bias=self.deconv_with_bias))
            # layers.append(nn.BatchNorm2d(self.inplanes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))

            layers.append(nn.Conv2d(self.inplanes, planes, 1, 1, 0, bias=True)) # pointwise convolution
            # layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))

            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()] 
        #x = self.conv1(x)
        x = self.features(x)

        x = self.conv2(x)

        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0) # no bias

            # pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            # self.load_state_dict(pretrained_state_dict, strict=False)
            checkpoint = torch.load(pretrained)

            if isinstance(checkpoint, OrderedDict):
                state_dict = OrderedDict()
                state_dict = checkpoint
                # when size mismatched 
                '''for key in checkpoint.keys():
                    #'collections.OrderedDict' object has no attribute 'state_dict'
                    #print(key)
                    #print(checkpoint[key].size())
                    #print(eval('self.'+key+'.size()'))
                    if(re.match(r'features.[2-4].conv.[0-7].weight',key)!=None):
                        print('0:',key,checkpoint[key].size())
                        #print('model',self.state_dict()[key].size())
                        #print('checkpoint:',checkpoint[key])
                        state_dict[key] = torch.ones(checkpoint[key].size())
                    elif(re.match(r'features.[2-4].',key)!=None):
                        print('1:',key,checkpoint[key].size())
                        state_dict[key] = torch.zeros(checkpoint[key].size())
                    else:
                        state_dict[key] = checkpoint[key]'''

            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict_old = checkpoint['state_dict']
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith('module.'):
                        # state_dict[key[7:]] = state_dict[key]
                        # state_dict.pop(key)
                        state_dict[key[7:]] = state_dict_old[key]
                    else:
                        state_dict[key] = state_dict_old[key]
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                print(type(checkpoint))
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(pretrained))

            self.load_state_dict(state_dict, strict=False) #strict-False 所以可以将量化后的权重导入！
        else:
            logger.error('=> imagenet pretrained model dose not exist')
            logger.error('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


def get_pose_net(cfg, stages_repeats, stages_out_channels,  is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    #style = cfg.MODEL.STYLE  # deprecated

    #block_class, layers = resnet_spec[num_layers]
    block_class = ""
    layers = []

    model = PoseMobileNet(block_class, layers, cfg, stages_repeats, stages_out_channels, **kwargs)
    #model = PoseShuffleNet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model