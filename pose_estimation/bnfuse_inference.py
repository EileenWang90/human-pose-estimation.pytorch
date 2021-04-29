#####################################################################
#以下函数用于将量化感知训练的bn_fuse的权重数据进行融合，即变成整合成conv，生成只用于推断的权重文件
# Written by Yiting Wang 2021/04/21
#####################################################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torchsummary import summary

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

import dataset
import models

import quantize_dorefa
from quantize_iao import *
# from quantize_iao_uint import *  #对feature map进行uint对称量化

def bn_fuse_conv(bn_conv,device):
    # ******************** bn参数 *********************
    mean = bn_conv.running_mean
    std = torch.sqrt(bn_conv.running_var + bn_conv.eps)
    gamma = bn_conv.gamma
    beta = bn_conv.beta
    # ******************* conv参数 ********************
    w = bn_conv.weight
    w_fused = w.clone()
    if bn_conv.bias is not None:
        b = bn_conv.bias
    else:
        b = mean.new_zeros(mean.shape)
    b_fused = b.clone()
    # ******************* bn融合 *******************
    w_fused = w * (gamma / std).reshape([bn_conv.out_channels, 1, 1, 1])
    b_fused = beta + (b - mean) * (gamma / std)
    bn_fused_conv = QuantConv2d(bn_conv.in_channels,
                                         bn_conv.out_channels,
                                         bn_conv.kernel_size,
                                         stride=bn_conv.stride,
                                         padding=bn_conv.padding,
                                         dilation=bn_conv.dilation,
                                         groups=bn_conv.groups,
                                         bias=True,
                                         padding_mode=bn_conv.padding_mode,
                                         a_bits=config.QUANTIZATION.A_BITS,
                                         w_bits=config.QUANTIZATION.W_BITS,
                                         q_type=config.QUANTIZATION.Q_TYPE,
                                         q_level=config.QUANTIZATION.Q_LEVEL,
                                         device=device,
                                         quant_inference=True)
    bn_fused_conv.weight.data = w_fused
    bn_fused_conv.bias.data = b_fused
    bn_fused_conv.activation_quantizer.scale.copy_(bn_conv.activation_quantizer.scale)
    bn_fused_conv.activation_quantizer.zero_point.copy_(bn_conv.activation_quantizer.zero_point)
    bn_fused_conv.activation_quantizer.eps = bn_conv.activation_quantizer.eps
    bn_fused_conv.weight_quantizer.scale.copy_(bn_conv.weight_quantizer.scale)
    bn_fused_conv.weight_quantizer.zero_point.copy_(bn_conv.weight_quantizer.zero_point)
    bn_fused_conv.weight_quantizer.eps = bn_conv.weight_quantizer.eps
    return bn_fused_conv

def bn_fuse_deconv(bn_conv,device):
    # ******************** bn参数 *********************
    mean = bn_conv.running_mean
    std = torch.sqrt(bn_conv.running_var + bn_conv.eps)
    gamma = bn_conv.gamma
    beta = bn_conv.beta
    # ******************* conv参数 ********************
    w = bn_conv.weight
    w_fused = w.clone()
    if bn_conv.bias is not None:
        b = bn_conv.bias
    else:
        b = mean.new_zeros(mean.shape)
    b_fused = b.clone()
    # ******************* bn融合 *******************
    w_fused = w * (gamma / std).reshape([bn_conv.out_channels, 1, 1, 1])
    b_fused = beta + (b - mean) * (gamma / std)
    bn_fused_conv = QuantConvTranspose2d(bn_conv.in_channels,
                                         bn_conv.out_channels,
                                         bn_conv.kernel_size,
                                         stride=bn_conv.stride,
                                         padding=bn_conv.padding,
                                         output_padding=bn_conv.output_padding,
                                         dilation=bn_conv.dilation,
                                         groups=bn_conv.groups,
                                         bias=True,
                                         padding_mode=bn_conv.padding_mode,
                                         a_bits=config.QUANTIZATION.A_BITS,
                                         w_bits=config.QUANTIZATION.W_BITS,
                                         q_type=config.QUANTIZATION.Q_TYPE,
                                         q_level=config.QUANTIZATION.Q_LEVEL,
                                         device=device,
                                         quant_inference=True)
    bn_fused_conv.weight.data = w_fused
    bn_fused_conv.bias.data = b_fused
    bn_fused_conv.activation_quantizer.scale.copy_(bn_conv.activation_quantizer.scale)
    bn_fused_conv.activation_quantizer.zero_point.copy_(bn_conv.activation_quantizer.zero_point)
    bn_fused_conv.activation_quantizer.eps = bn_conv.activation_quantizer.eps
    bn_fused_conv.weight_quantizer.scale.copy_(bn_conv.weight_quantizer.scale)
    bn_fused_conv.weight_quantizer.zero_point.copy_(bn_conv.weight_quantizer.zero_point)
    bn_fused_conv.weight_quantizer.eps = bn_conv.weight_quantizer.eps
    return bn_fused_conv


def bn_fuse_module(module, device):
    for name, child in module.named_children():
        if isinstance(child, QuantBNFuseConv2d):
            bn_fused_conv = bn_fuse_conv(child, device)
            module._modules[name] = bn_fused_conv
        elif isinstance(child, QuantBNFuseConvTranspose2d):
            bn_fused_deconv = bn_fuse_deconv(child, device)
            module._modules[name] = bn_fused_deconv
        else:
            bn_fuse_module(child, device)


def model_bn_fuse(model, inplace=False):
    if not inplace:
        model = copy.deepcopy(model)
    device = next(model.parameters()).device
    bn_fuse_module(model,device)
    return model

def select_device(device='', apex=False, batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        # os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA ' + ('Apex ' if apex else '')  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='experiments/coco/resnet50/mobile_quant_relu_int.yaml',
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)
    parser.add_argument('--model-file',
                        help='model state file',
                        type=str)
    parser.add_argument('--use-detect-bbox',
                        help='use detect bbox',
                        action='store_true')
    parser.add_argument('--flip-test',
                        help='use flip test',
                        action='store_true')
    parser.add_argument('--post-process',
                        help='use post process',
                        action='store_true')
    parser.add_argument('--shift-heatmap',
                        help='shift heatmap',
                        action='store_true')
    parser.add_argument('--coco-bbox-file',
                        help='coco detection bbox file',
                        type=str)
    parser.add_argument('--w_bits', type=int, help='the bit of weights you want to quantize')     # W — bits   要设置一致呀，真的是血的教训
    parser.add_argument('--a_bits', type=int, help='the bit of feature map you want to quantize')     # A — bits
    parser.add_argument('--bn_fuse', type=int, help='bn_fuse:0,1')     # bn融合标志位
    parser.add_argument('--q_type', type=int, help='quant_type:0-symmetric, 1-asymmetric')     # 量化方法选择
    parser.add_argument('--q_level', type=int, help='quant_level:0-per_channel, 1-per_layer')     # 量化级别选择
    parser.add_argument('--weight_observer', type=int, help='quant_weight_observer:0-MinMaxObserver, 1-MovingAverageMinMaxObserver')     # weight_observer选择
    parser.add_argument('--quant_inference', action='store_false', help='default quant_inference True')  #是否进行量化推断
    parser.add_argument('--quant_method', type=int, help='quant_method:0-IAO, 1-DoReFa')  #使用什么量化方法
    
    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.use_detect_bbox:
        config.TEST.USE_GT_BBOX = not args.use_detect_bbox
    if args.flip_test:
        config.TEST.FLIP_TEST = args.flip_test
    if args.post_process:
        config.TEST.POST_PROCESS = args.post_process
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
    if args.coco_bbox_file:
        config.TEST.COCO_BBOX_FILE = args.coco_bbox_file
    #quantization
    if args.w_bits:
        config.QUANTIZATION.W_BITS = args.w_bits
    if args.a_bits:
        config.QUANTIZATION.A_BITS = args.a_bits
    if args.bn_fuse:
        config.QUANTIZATION.BN_FUSE = args.bn_fuse
    if args.q_type:
        config.QUANTIZATION.Q_TYPE = args.q_type
    if args.q_level:
        config.QUANTIZATION.Q_LEVEL = args.q_level
    if args.weight_observer:
        config.QUANTIZATION.WEIGHT_OBSERVER = args.weight_observer
    if args.quant_inference:
        config.QUANTIZATION.QUANT_INFERENCE = args.quant_inference
    if args.quant_method:
        config.QUANTIZATION.QUANT_METHOD = args.quant_method


def main():
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # for shufflenetv2
    shufflenetv2_spec = {'0.5': ([4, 8, 4], [24, 48, 96, 192, 1024]),
                         '1.0': ([4, 8, 4], [24, 116, 232, 464, 1024]),
                         '1.5': ([4, 8, 4], [24, 176, 352, 704, 1024]),
                         '2.0': ([4, 8, 4], [24, 244, 488, 976, 2048])}
    stages_repeats, stages_out_channels = shufflenetv2_spec['1.0']
    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, 
        stages_repeats, stages_out_channels,
        is_train=False
    )
    gpus = [int(i) for i in config.GPUS.split(',')]
    device = select_device(config.GPUS, batch_size=config.TEST.BATCH_SIZE*len(gpus))

    model = model.to(device)
    # summary(model,input_size=(3, 256, 192))

    #print('*******************ori_model*******************\n', model)
    if(config.QUANTIZATION.QUANT_METHOD == 1): # DoReFa
        quantize_dorefa.prepare(model, inplace=True, a_bits=config.QUANTIZATION.A_BITS, w_bits=config.QUANTIZATION.W_BITS, quant_inference=config.QUANTIZATION.QUANT_INFERENCE, is_activate=False)
    else: #default quant_method == 0   IAO
        prepare(model, inplace=True, a_bits=config.QUANTIZATION.A_BITS, w_bits=config.QUANTIZATION.W_BITS,q_type=config.QUANTIZATION.Q_TYPE, q_level=config.QUANTIZATION.Q_LEVEL, device=device,#device=next(model.parameters()).device, 
                            weight_observer=config.QUANTIZATION.WEIGHT_OBSERVER, bn_fuse=config.QUANTIZATION.BN_FUSE, quant_inference=config.QUANTIZATION.QUANT_INFERENCE)
    #print('\n*******************quant_model*******************\n', model)
    print('\n*******************Using quant_model in test*******************\n')
    
    # if config.TEST.MODEL_FILE:
    #     logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
    #     if(config.TEST.MODEL_FILE.split('/')[-1]=='checkpoint.pth.tar'):
    #         model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    #         #model.load_state_dict(torch.load(config.TEST.MODEL_FILE,map_location=torch.device('cuda'))['state_dict'])
    #         model.load_state_dict(torch.load(config.TEST.MODEL_FILE,map_location=device)['state_dict'])
    #         #torch.save(model.module.state_dict(), 'output/coco_quan/mobile_quant_relu_w8a8_bnfuse0/checkpoint_nomodule.pth.tar')
    #     elif(config.TEST.MODEL_FILE.split('/')[-1]=='model_best.pth.tar'):  #multiGPU has model.module.
    #         model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    #         model.load_state_dict(torch.load(config.TEST.MODEL_FILE,map_location=device))
    #     elif(config.TEST.MODEL_FILE.split('/')[-1]=='checkpoint_resave.pth.tar'):  #multiGPU has model.module.
    #         model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    #         model.load_state_dict(torch.load(config.TEST.MODEL_FILE,map_location=device))
    #     else:  #final_state.pth.tar
    #         model.load_state_dict(torch.load(config.TEST.MODEL_FILE,map_location=device))
    #         model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    # else:
    #     model_state_file = os.path.join(final_output_dir,
    #                                     'final_state.pth.tar')
    #     logger.info('=> loading model from {}'.format(model_state_file))
    #     #print('0:',next(model.parameters()).device) #查看模型在CPU还是GPU上  cpu
    #     model.load_state_dict(torch.load(model_state_file, map_location=device),strict=False)
    #     #print('1:',next(model.parameters()).device) #查看模型在CPU还是GPU上  cpu
    #     model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    #     #print('2:',next(model.parameters()).device) #查看模型在CPU还是GPU上  cuda:0

    model.to(device)
    # ********************* quant_bn_fused_model_inference **********************
    model_bn_fuse(model, inplace=True)  # bn融合
    # print('\n*******************For inference bn_fuse quant_model*******************\n', model)
    ckpt = {'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()}
    # torch.save(ckpt, 'output/weights_quan/int8_mobilenet8_relu_bnfuse_inference.pt')

    logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
    model.load_state_dict(torch.load(config.TEST.MODEL_FILE,map_location=device)['model'])  ##为什么还在'model'里面呀？
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # model.to(device)
    # summary(model,input_size=(3, 256, 192))
    model.to(device)

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    valid_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    print('validate:',next(model.parameters()).device) #查看模型在CPU还是GPU上  
    # evaluate on validation set
    validate(config, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)


if __name__ == '__main__':
    main()