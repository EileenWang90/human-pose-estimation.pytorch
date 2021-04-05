import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ********************* quantizers（量化器，量化） *********************
# 取整(ste)
class Round(Function):
    @staticmethod
    def forward(self, input):
        output = torch.round(input) #Returns a new tensor with each of the elements of input rounded to the closest integer.
        return output

    @staticmethod  #在构建模型时，遇到不可求导的操作需要自定义求导方式
    def backward(self, grad_output): #grad_output：存储forward后tensor的梯度，是反向传播上一级计算得到的梯度值
        grad_input = grad_output.clone() #自定义梯度计算，不计算取证计算的梯度
        return grad_input

# A(特征)量化
class ActivationQuantizer(nn.Module):
    def __init__(self, a_bits):
        super(ActivationQuantizer, self).__init__()
        self.a_bits = a_bits

    # 取整(ste)
    def round(self, input):
        output = Round.apply(input)
        return output

    # 量化/反量化
    def forward(self, input):
        '''
        if self.a_bits == 32:
            output = input
        elif self.a_bits == 1:'''
        if self.a_bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.a_bits != 1
        else:
            output = torch.clamp(input * 0.1, 0, 1)      # 特征A截断前先进行缩放（* 0.1），以减小截断误差  会引入误差嘛？相当于已经进行relu了?似乎并不是
            #output = torch.clamp(input, 0, 1)      # 特征A截断前先进行缩放（* 0.1），以减小截断误差  会引入误差嘛？相当于已经进行relu了?似乎并不是
            scale = 1 / float(2 ** self.a_bits - 1)      # scale
            output = self.round(output / scale) * scale  # 量化/反量化  [0,1]
        return output

# W(权重)量化
class WeightQuantizer(nn.Module):
    def __init__(self, w_bits):
        super(WeightQuantizer, self).__init__()
        self.w_bits = w_bits

    # 取整(ste)
    def round(self, input):
        output = Round.apply(input)
        return output

    # 量化/反量化
    def forward(self, input):
        '''
        if self.w_bits == 32:
            output = input
        elif self.w_bits == 1:'''
        if self.w_bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.w_bits != 1
        else:
            #print(input.shape) #[8,8,1,1]
            #print(input[])
            output = torch.tanh(input)  # scale to [-1,1]
            output = output / 2 / torch.max(torch.abs(output)) + 0.5  # 归一化-[0,1] 对称量化
            scale = 1 / float(2 ** self.w_bits - 1)                   # scale
            output = self.round(output / scale) * scale               # 量化/反量化
            output = 2 * output - 1  #[0,1]->[-1,1]
            #print(output)
            
            '''
            err = output - input
            err = torch.abs(err)
            max = err.max()
            if (max > 1) :
                D1 = torch.max(err, 1)
                D2 = torch.max(D1[0], 1)
                D3 = torch.max(D2[0], 1)
                print(D1[1])
                print("#################################################################\n")
                print(D2[1])
                print("===================================================================\n")
                print(D3[1])
                print("=================================0000000000==================================\n")
                print(input)
                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
                print(output)
                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
                print(err)
                exit()'''

        return output


class QuantConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 a_bits=8,
                 w_bits=8,
                 quant_inference=False,
                 is_activate=True):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias, padding_mode)
        self.quant_inference = quant_inference
        self.is_activate = is_activate
        self.activation_quantizer = ActivationQuantizer(a_bits=a_bits)
        self.weight_quantizer = WeightQuantizer(w_bits=w_bits)

    def forward(self, input):
        if(self.is_activate):
            quant_input = self.activation_quantizer(input)
        else:
            quant_input = input  
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.conv2d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.dilation,
                          self.groups)
        return output


class QuantConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 a_bits=8,
                 w_bits=8,
                 quant_inference=False,
                 is_activate=True):
        super(QuantConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding,
                                                   dilation, groups, bias, padding_mode)
        self.quant_inference = quant_inference
        self.is_activate = is_activate
        self.activation_quantizer = ActivationQuantizer(a_bits=a_bits)
        self.weight_quantizer = WeightQuantizer(w_bits=w_bits)

    def forward(self, input):
        if(self.is_activate):
            quant_input = self.activation_quantizer(input)
        else:
            quant_input = input
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.conv_transpose2d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.output_padding,
                                    self.groups, self.dilation)
        return output


class QuantLinear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 a_bits=8,
                 w_bits=8,
                 quant_inference=False,
                 is_activate=True):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.quant_inference = quant_inference
        self.is_activate = is_activate
        self.activation_quantizer = ActivationQuantizer(a_bits=a_bits)
        self.weight_quantizer = WeightQuantizer(w_bits=w_bits)

    def forward(self, input):
        if(self.is_activate):
            quant_input = self.activation_quantizer(input)
        else:
            quant_input = input
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.linear(quant_input, quant_weight, self.bias)
        return output


def add_quant_op(module, layer_counter, a_bits=8, w_bits=8,
                 quant_inference=False, is_activate=True):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            layer_counter[0] += 1
            if layer_counter[0] > 1:
                if child.bias is not None:
                    quant_conv = QuantConv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride,
                                             padding=child.padding, dilation=child.dilation,
                                             groups=child.groups, bias=True, padding_mode=child.padding_mode,
                                             a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference,
                                             is_activate=is_activate)
                    quant_conv.bias.data = child.bias
                else:
                    quant_conv = QuantConv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride,
                                             padding=child.padding, dilation=child.dilation,
                                             groups=child.groups, bias=False, padding_mode=child.padding_mode,
                                             a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference,
                                             is_activate=is_activate)
                quant_conv.weight.data = child.weight
                module._modules[name] = quant_conv
        elif isinstance(child, nn.ConvTranspose2d):
            layer_counter[0] += 1
            if layer_counter[0] > 1:
                if child.bias is not None:
                    quant_conv_transpose = QuantConvTranspose2d(child.in_channels,
                                                                child.out_channels,
                                                                child.kernel_size,
                                                                stride=child.stride,
                                                                padding=child.padding,
                                                                output_padding=child.output_padding,
                                                                dilation=child.dilation,
                                                                groups=child.groups,
                                                                bias=True,
                                                                padding_mode=child.padding_mode,
                                                                a_bits=a_bits,
                                                                w_bits=w_bits,
                                                                quant_inference=quant_inference,
                                                                is_activate=is_activate)
                    quant_conv_transpose.bias.data = child.bias
                else:
                    quant_conv_transpose = QuantConvTranspose2d(child.in_channels,
                                                                child.out_channels,
                                                                child.kernel_size,
                                                                stride=child.stride,
                                                                padding=child.padding,
                                                                output_padding=child.output_padding,
                                                                dilation=child.dilation,
                                                                groups=child.groups, bias=False,
                                                                padding_mode=child.padding_mode,
                                                                a_bits=a_bits,
                                                                w_bits=w_bits,
                                                                quant_inference=quant_inference,
                                                                is_activate=is_activate)
                quant_conv_transpose.weight.data = child.weight
                module._modules[name] = quant_conv_transpose
        elif isinstance(child, nn.Linear):
            layer_counter[0] += 1
            if layer_counter[0] > 1:
                if child.bias is not None:
                    quant_linear = QuantLinear(child.in_features, child.out_features,
                                               bias=True, a_bits=a_bits, w_bits=w_bits,
                                               quant_inference=quant_inference,
                                               is_activate=is_activate)
                    quant_linear.bias.data = child.bias
                else:
                    quant_linear = QuantLinear(child.in_features, child.out_features,
                                               bias=False, a_bits=a_bits, w_bits=w_bits,
                                               quant_inference=quant_inference,
                                               is_activate=is_activate)
                quant_linear.weight.data = child.weight
                module._modules[name] = quant_linear
        else:
            add_quant_op(child, layer_counter, a_bits=a_bits, w_bits=w_bits,
                         quant_inference=quant_inference, is_activate=is_activate)


def prepare(model, inplace=False, a_bits=8, w_bits=8, quant_inference=False, is_activate=True):
    if not inplace:
        model = copy.deepcopy(model)
    layer_counter = [0]
    add_quant_op(model, layer_counter, a_bits=a_bits, w_bits=w_bits,
                 quant_inference=quant_inference, is_activate=is_activate)
    return model
