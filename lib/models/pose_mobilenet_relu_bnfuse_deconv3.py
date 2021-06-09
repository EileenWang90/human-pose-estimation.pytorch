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
import numpy as np

############################## int推断时进行数据调整 #########################
# 先使用浮点值进行数据验算
convert_path='/home/ytwang/wyt_workspace/quantization/human-pose-estimation.pytorch/output/weights_quan_deconv3/'
# M_list0 = np.load(convert_path+'M_refactor.npy', allow_pickle=True) #量化感知训练得到的scale
# M_list = np.load(convert_path+'mscale_norelu_quant.npy', allow_pickle=True)  #去掉relu的量化反量化 得到的scale
# M_list2 = np.load(convert_path+'M0_quant_requant.npy', allow_pickle=True)  #MO量化反量化 得到的scale
BIT=16
# M0_list = np.load(convert_path+'M0_int.npy', allow_pickle=True)  #MO量化反量化 得到的scale   M0_int.npy(默认是16bit的)
M0_list = np.load(convert_path+'M0_int_shortcut0.npy', allow_pickle=True)  #MO量化反量化 得到的scale

wscale_list = np.load(convert_path+'wscale.npy', allow_pickle=True) 
# oscale_list = np.load(convert_path+'oscale.npy', allow_pickle=True)
oscale_list = np.load('/home/ytwang/wyt_workspace/quantization/human-pose-estimation.pytorch/output/weights_quan/oscale.npy', allow_pickle=True)

# ascale_list = np.load(convert_path+'ascale.npy', allow_pickle=True)
ascale_list = np.load(convert_path+'ascale_shortcut0.npy', allow_pickle=True)
Mkey_load = list(np.load(convert_path+'M_key.npy', allow_pickle=True))    #type:np.ndarray ->list  是59层名称的列表
# Mkey_load=list(Mkey_load)
verbose=False

def int_adjust(data, Mkey, adjust=False):  #包括层量化和通道量化
    # print(M_list.item()[Mkey].shape, Mkey) #, M_list.item()[Mkey]) # torch.Size([16]) 
    # print(data.shape) #conv1 torch.Size([128, 16, 128, 96])
    result_path='output/weights_quan_deconv3/validate_shortcut0/' 
    # result_path='output/weights_quan/validate_norelurequant/' 
    # result_path='output/weights_quan/validate_M0/' 
    # result_path='output/weights_quan/validate_M0_round/' 
    if(adjust==True and Mkey=='final_layer'): #最后一层直接进行浮点计算 不需要舍入和截断
        # print('It is final layer in intmodel.')
        # if(verbose==True):
        if(1):
            data_not_xM0=data[0].detach().cpu().numpy().transpose(1,2,0).reshape(-1,data[0].shape[0])
            np.savetxt(result_path+Mkey+'_int_output_not_xM0.txt', data_not_xM0, fmt="%d", delimiter='  ') 
        data = data * M0_list.item()[Mkey].to(data.device) #.type(torch.int32) # torch.clamp(x, qmin, qmax) w8a8
        print(M0_list.item()[Mkey])
        # tmp=data[0].detach().cpu().numpy().reshape(data[0].shape[0],-1)
        if(verbose==True):
            tmp=data[0].detach().cpu().numpy().transpose(1,2,0).reshape(data[0].shape[0],-1)
            np.savetxt(result_path+'final_layer_output.txt', tmp, fmt="%f", delimiter='  ') 
        # print(time)

    elif(adjust==True): #如果进行int计算，则需要*M并截断操作； 否则不对数据进行处理
        # Mkey_post = Mkey_load[Mkey_load.index(Mkey)+1] #后一层的ascale
        # #################### shortcut处直接使用out部分的scale ##################
        # shortcut_flag=False
        # for i in shortcut_list:
        #     if(i==Mkey):
        #         shortcut_flag=True
        # if(shortcut_flag==True):
        #     Mkey_shortcut = Mkey_load[Mkey_load.index(Mkey)+3]
        #     Mscale = wscale_list.item()[Mkey]*ascale_list.item()[Mkey_shortcut]/ascale_list.item()[Mkey_post]
        # else:
        #     Mscale = wscale_list.item()[Mkey]*ascale_list.item()[Mkey]/ascale_list.item()[Mkey_post]
        #直接使用下一层的ascale作为本层的oscale
        # Mkey_post = Mkey_load[Mkey_load.index(Mkey)+1] #后一层的ascale
        # Mscale = wscale_list.item()[Mkey]*ascale_list.item()[Mkey]/ascale_list.item()[Mkey_post]
        # print(Mkey,Mkey_post)

        # print("Mscale:", Mscale.flatten(),"\nM_list[]=",M_list.item()[Mkey].flatten())
        # data = torch.round(data * Mscale.to(data.device)).clamp_(-128, 127)#.type(torch.int32) # torch.clamp(x, qmin, qmax) w8a8
        #使用relu前对featuremap进行量化的scale作为oscale
        # print(data[0][0][0][0:2],data[0][0][1][0:2])
        if(verbose==True):
            data_not_xM0=data[0].detach().cpu().numpy().transpose(1,2,0).reshape(-1,data[0].shape[0])
            np.savetxt(result_path+Mkey+'_int_output_not_xM0.txt', data_not_xM0, fmt="%d", delimiter='  ') 
        # data = torch.round(data * M_list.item()[Mkey].to(data.device)).clamp_(-128, 127)#.type(torch.int32) # torch.clamp(x, qmin, qmax) w8a8
        data = ((data * M0_list.item()[Mkey].to(data.device) + 2**(BIT-1)).type(torch.int32)>>(BIT)).clamp_(-128, 127).type(torch.float32)#.type(torch.int32) # *M0并移位
        # print(data[0][0][0][0:2],data[0][0][1][0:2])
        # print(M0_list.item()[Mkey].flatten(),'\n',M_list1.item()[Mkey].flatten(),'\n')

        # Mkey_1=Mkey_load[Mkey_load.index(Mkey)+1] #后一层的ascale
        # M_true=ascale_list.item()[Mkey]*wscale_list.item()[Mkey]/ascale_list.item()[Mkey_1]
        # print("M_true:",M_true,ascale_list.item()[Mkey], wscale_list.item()[Mkey], ascale_list.item()[Mkey_1])
        # data = torch.round(data * M_true.to(data.device)).clamp_(-128, 127)#.type(torch.int32) # torch.clamp(x, qmin, qmax) w8a8

        if(verbose==True):
            # tmp=data[0].detach().cpu().numpy().reshape(data[0].shape[0],-1)
            tmp=data[0].detach().cpu().numpy().transpose(1,2,0).reshape(-1,data[0].shape[0])
            # print(data.shape, tmp.shape)
            np.savetxt(result_path+Mkey+'_int_output.txt', tmp, fmt="%d", delimiter='  ') 
    else: #浮点模型推断
        result_path='output/weights_quan/validate_float/'
        # # imgs.astype(np.float32).tofile(result_path+'input000001_352x256.bin')
        # # imgs=imgs.reshape([-1,imgs.shape[-1]]) #(352*256,3)
        # # if(Mkey=='conv1'):
        # # featuremap=data[0].detach().cpu().numpy().reshape(data[0].shape[0],-1) #torch.Size([128, 16, 128, 96])
        # featuremap=data[0].detach().cpu().numpy().transpose(1,2,0).reshape(-1,data[0].shape[0]) #torch.Size([128, 16, 128, 96]) 每行是通道数
        # print(Mkey,"_oscale:",oscale_list.item()[Mkey])
        # qfeaturemap=torch.round(data.detach().cpu()/oscale_list.item()[Mkey]).clamp_(-128,127).type(torch.int8)
        # # print(torch.max(qfeaturemap),torch.min(qfeaturemap),torch.mean(qfeaturemap))
        # # qfeaturemap=qfeaturemap[0].numpy().reshape(data[0].shape[0],-1) #[16,128,96]->[16,128*96]=[16,12288]
        # qfeaturemap=qfeaturemap[0].numpy().transpose(1,2,0).reshape(-1,data[0].shape[0]) #[16,128,96]->[128,96,16]->[12288,16]
        # # print(qfeaturemap.shape)
        # np.savetxt(result_path+Mkey+'_output.txt', featuremap, fmt="%f", delimiter='  ') #一个batch中的第一张图片
        # np.savetxt(result_path+Mkey+'_qoutput.txt', qfeaturemap, fmt="%d", delimiter='  ') 
        # # if(data.shape[0]==80):
        # #     np.savetxt(result_path+Mkey+'_qoutput.txt', qfeaturemap, fmt="%d", delimiter='  ') 
    return data


def shortcut_adjust(data, Mkey, adjust=False):  #针对shortcut进行处理 2, 4,5, 7,8,9, 11,12, 14,15    [1,2,3,4,3,3,1]
    # tmp=Mkey.split('.')
    # tmp[1]=str(int(tmp[1])-1)
    # Mkey_pre='.'.join(tmp)
    # print(M_list.item()[Mkey].shape, Mkey,'    ',M_list.item()[Mkey_pre].shape, Mkey_pre) #, M_list.item()[Mkey]) # torch.Size([16]) 
    # print(data.shape) #conv1 torch.Size([128, 16, 128, 96])
    # scale = M_list.item()[Mkey] / M_list.item()[Mkey_pre]
    Mkey_pre = Mkey_load[Mkey_load.index(Mkey)-2] #往前数两层的oscale (一个InvertedResidual的一开始输入) x->conv_relu->conv_relu->conv
    # print(Mkey_pre, ascale_list.item()[Mkey_pre], oscale_list.item()[Mkey], scale)
    result_path='output/weights_quan_deconv3/validate_shortcut0/'
    # result_path='output/weights_quan/validate_M0_round/' 

    shortcut_quan_requant=False
    if(adjust==True and shortcut_quan_requant==True): #如果进行int计算，则需要*M并截断操作； 否则不对数据进行处理
        scale = ascale_list.item()[Mkey_pre] / oscale_list.item()[Mkey] #这儿的oscale_list.item()[Mkey]等于下一个卷积层的ascale
        data = torch.round(data * scale.to(data.device)).clamp_(-128, 127)#.type(torch.int32) # torch.clamp(x, qmin, qmax) w8a8
        # tmp=data[0].detach().cpu().numpy().reshape(data[0].shape[0],-1)
    if(verbose==True):
        tmp=data[0].detach().cpu().numpy().transpose(1,2,0).reshape(-1,data[0].shape[0])
        np.savetxt(result_path+Mkey+'_qx_shortcut.txt', tmp, fmt="%d", delimiter='  ') 
    return data


def save_quantize_results(data, Mkey, adjust=False):  #now it is for featuremap which will go into conv for calculation 得到进入conv计算后的整型feature map
    result_path='output/weights_quan_deconv3/validate_shortcut0/'
    # result_path='output/weights_quan/validate_norelurequant/'  
    # result_path='output/weights_quan/validate_M0_round/' 
    using_relu_oscale=False #如果relu前的featuremap进行了量化反量化，则需要进行scale转换 True; 否则为False
    if(adjust==True and using_relu_oscale): #如果进行int计算，则将feature map的整型结果保存
        Mkey_1=Mkey_load[Mkey_load.index(Mkey)-1] #前一层的oscale  
        # if(Mkey_1 != 'features.0.conv3'):
        # print(Mkey_1,"_oscale:",oscale_list.item()[Mkey_1], Mkey,"_ascale:",ascale_list.item()[Mkey])
        data = torch.round(data * oscale_list.item()[Mkey_1].to(data.device) /ascale_list.item()[Mkey].to(data.device)).clamp_(-128, 127)#.type(torch.int32) # torch.clamp(x, qmin, qmax) w8a8
        # # tmp=data[0].detach().cpu().numpy().reshape(data[0].shape[0],-1)
        # tmp=data[0].detach().cpu().numpy().transpose(1,2,0).reshape(-1,data[0].shape[0])
        # np.savetxt(result_path+Mkey+'_int_featuremap.txt', tmp, fmt="%d", delimiter='  ') 
    if(verbose==True):
        tmp=data[0].detach().cpu().numpy().transpose(1,2,0).reshape(-1,data[0].shape[0]) #
        np.savetxt(result_path+Mkey+'_int_featuremap.txt', tmp, fmt="%d", delimiter='  ') 
        if(Mkey=='conv2'):
            tmp.astype(np.int8).tofile('output/weights_quan_deconv3/'+'conv2_8x6x128.bin') #先通道，再行再列
    return data


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
    def __init__(self, inp, oup, stride, expand_ratio, count, int_adjust=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        #整形运算相关 默认使用浮点数进行计算
        self.int_adjust=int_adjust
        self.Mkey = 'features.' + str(count) #使用count作为键值索引对应的M
        self.expand_ratio = expand_ratio

        # if expand_ratio == 1:
        #     self.conv1 = nn.Sequential(
        #         # dw
        #         nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=True),
        #         # nn.BatchNorm2d(hidden_dim),
        #         nn.ReLU(inplace=True)
        #     )
        #     self.conv2 = nn.Sequential(
        #         # pw-linear
        #         nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=True),
        #         # nn.BatchNorm2d(oup),
        #     )
        #     self.conv3 = nn.Identity()  #这儿需要注意：直通，为了保持格式一致性
        # else:
        #     self.conv1 = nn.Sequential(
        #         # pw
        #         nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=True),
        #         # nn.BatchNorm2d(hidden_dim),
        #         nn.ReLU(inplace=True)
        #     )
        #     self.conv2 = nn.Sequential(
        #         # dw
        #         nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=True),
        #         # nn.BatchNorm2d(hidden_dim),
        #         nn.ReLU(inplace=True)
        #     )
        #     self.conv3 = nn.Sequential(
        #         # pw-linear
        #         nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=True),
        #         # nn.BatchNorm2d(oup),
        #     )
        self.relu = nn.ReLU(inplace=True)
        if expand_ratio == 1:
            self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=True) #dw
            self.conv2 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=True) # pw-linear
            self.conv3 = nn.Identity()  #这儿需要注意：直通，为了保持格式一致性
        else:
            self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=True) #pw
            self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=True) #dw
            self.conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=True) #pw_linear

    def forward(self, x):
        identity = x
        if self.use_res_connect: #stride=1且输入输出通道数相等
            x=save_quantize_results(x, self.Mkey+'.conv1', self.int_adjust)
            y=self.conv1(x)
            y=int_adjust(y, self.Mkey+'.conv1', self.int_adjust)
            y=self.relu(y)

            y=save_quantize_results(y, self.Mkey+'.conv2', self.int_adjust)
            y=self.conv2(y)
            y=int_adjust(y, self.Mkey+'.conv2', self.int_adjust)
            y=self.relu(y)

            y=save_quantize_results(y, self.Mkey+'.conv3', self.int_adjust)
            y=self.conv3(y)
            y=int_adjust(y, self.Mkey+'.conv3', self.int_adjust) #应该对shortcut之后的结果*M
            y= y + shortcut_adjust(identity, self.Mkey+'.conv3', self.int_adjust)  #这儿的identity也是需要进行转换的！
            return y
        else:
            x=save_quantize_results(x, self.Mkey+'.conv1', self.int_adjust)
            y=self.conv1(x)
            y=int_adjust(y, self.Mkey+'.conv1', self.int_adjust) #整形计算时对输出结果进行调整
            y=self.relu(y)

            y=save_quantize_results(y, self.Mkey+'.conv2', self.int_adjust)
            y=self.conv2(y)
            y=int_adjust(y, self.Mkey+'.conv2', self.int_adjust)
            if(self.expand_ratio!=1): #InvertedResidual最后一层是没有relu的
                y=self.relu(y)
                y=save_quantize_results(y, self.Mkey+'.conv3', self.int_adjust)
            
            y=self.conv3(y)
            if(self.expand_ratio!=1):
                y=int_adjust(y, self.Mkey+'.conv3', self.int_adjust) #InvertedResidual最后一层是没有relu的
            return y

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
        int_adjust=False,   #目前直接在这儿修改代码...
        **kwargs ) -> None:

        self.inplanes = 128
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        self.int_adjust = int_adjust

        super(PoseMobileNet, self).__init__()

        block = InvertedResidual
        input_channel = 32
        width_mult = 0.5

        input_channel = make_divisible((input_channel * width_mult)) # first channel is always 32!  ?
        # self.features = [conv_bn(3, input_channel, 2)]
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=True)
        self.features = []
        # building inverted residual blocks
        count = 0
        for t, c, n, s in interverted_residual_setting: # n: 模块重复次数
            output_channel = make_divisible((c * width_mult))
            #print("s=",s,"    output_channel=",output_channel)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t, count=count, int_adjust=self.int_adjust))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t, count=count, int_adjust=self.int_adjust))
                input_channel = output_channel
                count += 1
        # building last several layers
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # in order to make the channel consistent  channel{v2_1.0: 464 -> resnet self.inplanes: 128}
        self.conv2 = nn.Conv2d(input_channel, self.inplanes, kernel_size=1, stride=1, padding=0, bias=True)
     
        self.deconv_layers = []
        # used for deconv layers
        self.layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        # self.deconv_layers0=nn.Sequential(*(self.layers[0:2])) #nn.ConvTranspose2d+relu
        # self.deconv_layers1=nn.Sequential(*(self.layers[2:4])) #nn.Conv2d+relu
        # self.deconv_layers2=nn.Sequential(*(self.layers[4:6]))
        # self.deconv_layers3=nn.Sequential(*(self.layers[6:8]))
        # self.deconv_layers4=nn.Sequential(*(self.layers[8:10]))
        # self.deconv_layers5=nn.Sequential(*(self.layers[10:12]))
        self.deconv_layers0=self.layers[0] #nn.ConvTranspose2d
        self.deconv_layers1=self.layers[2] #nn.Conv2d
        self.deconv_layers2=self.layers[4]
        self.deconv_layers3=self.layers[6]
        self.deconv_layers4=self.layers[8]
        self.deconv_layers5=self.layers[10]


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
                    bias=True)) #bnfuse之后一定会有bias
                    #bias=self.deconv_with_bias))
            # layers.append(nn.BatchNorm2d(self.inplanes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))

            layers.append(nn.Conv2d(self.inplanes, planes, 1, 1, 0, bias=True)) # pointwise convolution
            # layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))

            self.inplanes = planes

        # return nn.Sequential(*layers)
        return layers

    def forward(self, x):
        # See note [TorchScript super()] 
        x = self.conv1(x)
        x=int_adjust(x, 'conv1', self.int_adjust)
        x=self.relu(x)

        x = self.features(x)

        x=save_quantize_results(x, 'conv2', self.int_adjust)
        x = self.conv2(x)
        x=int_adjust(x, 'conv2', self.int_adjust)
        x=self.relu(x)

        x=save_quantize_results(x, 'deconv_layers0', self.int_adjust)
        x = self.deconv_layers0(x)
        x=int_adjust(x, 'deconv_layers0', self.int_adjust)
        x=self.relu(x)

        x=save_quantize_results(x, 'deconv_layers1', self.int_adjust)
        x = self.deconv_layers1(x)
        x=int_adjust(x, 'deconv_layers1', self.int_adjust)
        x=self.relu(x)

        x=save_quantize_results(x, 'deconv_layers2', self.int_adjust)
        x = self.deconv_layers2(x)
        x=int_adjust(x, 'deconv_layers2', self.int_adjust)
        x=self.relu(x)

        x=save_quantize_results(x, 'deconv_layers3', self.int_adjust)
        x = self.deconv_layers3(x)
        x=int_adjust(x, 'deconv_layers3', self.int_adjust)
        x=self.relu(x)

        x=save_quantize_results(x, 'deconv_layers4', self.int_adjust)
        x = self.deconv_layers4(x)
        x=int_adjust(x, 'deconv_layers4', self.int_adjust)
        x=self.relu(x)

        x=save_quantize_results(x, 'deconv_layers5', self.int_adjust)
        x = self.deconv_layers5(x)
        x=int_adjust(x, 'deconv_layers5', self.int_adjust)
        x=self.relu(x)

        x=save_quantize_results(x, 'final_layer', self.int_adjust)
        x = self.final_layer(x)
        x=int_adjust(x, 'final_layer', self.int_adjust)
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

############################# 获取中间层数据结果 使用hook ################################
fmap_block = dict()  # 装feature map
def farward_hook(module, inp, outp):  #似乎并不需要，虽然使用hook可能会更加优雅...
    fmap_block['input'] = inp
    fmap_block['output'] = outp


def get_pose_net(cfg, stages_repeats, stages_out_channels,  is_train, int_adjust=False,  **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    #style = cfg.MODEL.STYLE  # deprecated

    #block_class, layers = resnet_spec[num_layers]
    block_class = ""
    layers = []

    model = PoseMobileNet(block_class, layers, cfg, stages_repeats, stages_out_channels, int_adjust=int_adjust, **kwargs)
    #model = PoseShuffleNet(cfg, **kwargs)

    # 注册hook
    model.conv2.register_forward_hook(farward_hook)
    # model.deconv_layers.register_forward_hook(farward_hook)
    # model.conv2.register_forward_hook(farward_hook)
    # print(len(fmap_block['input']))
    # print(fmap_block['input'][0].shape)
    # print(len(fmap_block['output']))
    # print(fmap_block['output'][0].shape)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model