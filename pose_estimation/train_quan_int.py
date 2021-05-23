# ------------------------------------------------------------------------------
# Written by Yiting Wang 2021/04/04
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torchsummary import summary

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.config import get_model_name
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger

import dataset
import models

import quantize_dorefa
from quantize_iao import *
# from quantize_iao_uint import *  #对feature map进行uint对称量化

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
                        default='experiments/coco/resnet50/mobile_quant_relu_int.yaml',  #experiments/coco/resnet50/mobile_quant_allrelu_int.yaml #量化结果是错误的
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
    parser.add_argument('--noresume',
                        help='not to train from previous model weights(default:resume)',
                        action='store_true')
    parser.add_argument('--w_bits', type=int, help='the bit of weights you want to quantize')     # W — bits
    parser.add_argument('--a_bits', type=int, help='the bit of feature map you want to quantize')     # A — bits
    parser.add_argument('--bn_fuse', type=int, help='bn_fuse:1')     # bn融合标志位
    parser.add_argument('--q_type', type=int, help='quant_type:0-symmetric, 1-asymmetric')     # 量化方法选择
    parser.add_argument('--q_level', type=int, help='quant_level:0-per_channel, 1-per_layer')     # 量化级别选择
    parser.add_argument('--weight_observer', type=int, help='quant_weight_observer:0-MinMaxObserver, 1-MovingAverageMinMaxObserver')     # weight_observer选择
    parser.add_argument('--quant_inference', action='store_true', help='default quant_inference False')  #是否进行量化推断 训练时候不使用量化推断
    parser.add_argument('--quant_method', type=int, help='quant_method:0-IAO, 1-DoReFa')  #使用什么量化方法 

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
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
        config, args.cfg, 'train')

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

    is_train = True  #则会使用posenet中的权重初始化函数
    if(~args.noresume):   #如果使用posenet中的权重初始化函数，则model.load_state_dict函数并没有使用map_location=device参数，多GPU跑的时候test是0的
        is_train = False

    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config,
        stages_repeats, stages_out_channels,
        is_train=is_train,
    )
    # print(model)
    #model = model.cuda()
    #summary(model,input_size=(3, 256, 192))
    gpus = [int(i) for i in config.GPUS.split(',')]
    device = select_device(config.GPUS, batch_size=config.TEST.BATCH_SIZE*len(gpus))
    #model = model.to(device)
    
    # ######################## 使用float版本的weight.pt时 ##################################   这时用两块GPU test结果还是0
    # if(~args.noresume):
    #     is_train = False
    #     model.load_state_dict(torch.load(config.MODEL.PRETRAINED, map_location=device))
    #     #model.load_state_dict(torch.load(config.MODEL.PRETRAINED))
    #     print('Load model weight from',config.MODEL.PRETRAINED)

    ################################## quantization model #################################
    # print('*******************ori_model*******************\n', model)
    if(config.QUANTIZATION.QUANT_METHOD == 1): # DoReFa
        quantize_dorefa.prepare(model, inplace=True, a_bits=config.QUANTIZATION.A_BITS, w_bits=config.QUANTIZATION.W_BITS, quant_inference=config.QUANTIZATION.QUANT_INFERENCE, is_activate=False)
    else: #default quant_method == 0   IAO
        prepare(model, inplace=True, a_bits=config.QUANTIZATION.A_BITS, w_bits=config.QUANTIZATION.W_BITS,q_type=config.QUANTIZATION.Q_TYPE, q_level=config.QUANTIZATION.Q_LEVEL, #device=device, 放了就会出错！ 
                            weight_observer=config.QUANTIZATION.WEIGHT_OBSERVER, bn_fuse=config.QUANTIZATION.BN_FUSE, quant_inference=config.QUANTIZATION.QUANT_INFERENCE)
    # print('\n*******************quant_model*******************\n', model)
    print('\n*******************Using quant_model in test*******************\n')

    ####################### 使用量化版本的weight.pt时 ##################################
    if(~args.noresume):
        is_train = False
        print(device) #cuda:0   device=torch.device('cuda:0')
        model.load_state_dict(torch.load(config.MODEL.PRETRAINED, map_location=device))
        #model.load_state_dict(torch.load(config.MODEL.PRETRAINED))
        print('Load model weight from',config.MODEL.PRETRAINED)

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', config.MODEL.NAME + '.py'),
        final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand((config.TRAIN.BATCH_SIZE,
                             3,
                             config.MODEL.IMAGE_SIZE[1],
                             config.MODEL.IMAGE_SIZE[0]))
    writer_dict['writer'].add_graph(model.to('cpu'), (dump_input, ), verbose=False)  #weights(model)和input必须类型一致，即必须都放在cpu或者gpu上

    #print('0:',next(model.parameters()).device) #查看模型在CPU还是GPU上  
    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    #print('1:',next(model.parameters()).device) #查看模型在CPU还是GPU上  

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    optimizer = get_optimizer(config, model)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
    )

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TRAIN_SET,
        True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
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

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    best_perf = 0.0
    best_model = False
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        lr_scheduler.step()

        # train for one epoch
        #print('train',next(model.parameters()).device) #查看模型在CPU还是GPU上  
        train(config, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)


        # evaluate on validation set
        #print('validate:',next(model.parameters()).device) #查看模型在CPU还是GPU上  
        perf_indicator = validate(config, valid_loader, valid_dataset, model,
                                  criterion, final_output_dir, tb_log_dir, writer_dict)

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': get_model_name(config),
            'state_dict': model.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
