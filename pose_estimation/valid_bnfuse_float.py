# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
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


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3_mobile.yaml',  #要将其中的deconv 修改成3或者4
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

    ###################################################################################
    int_adjust=False    #在这儿修改float 还是int型
    ###################################################################################

    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, 
        stages_repeats, stages_out_channels,
        is_train=False,
        int_adjust=int_adjust,
    )
    model = model.cuda()
    # summary(model,input_size=(3, 256, 192))
    # print(model)
    # for n,param_tensor in enumerate(model.state_dict()):
    #     #打印 key value字典
    #     print(n, param_tensor,'\t',model.state_dict()[param_tensor].size())

    ################################## int版本模型 #########################################
    int_model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, 
        stages_repeats, stages_out_channels,
        is_train=False,
        int_adjust=True,
    )
    int_model = int_model.cuda()
    ########################################################################################

    # if config.TEST.MODEL_FILE:
    #     logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
    #     model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    # else:
    #     model_state_file = os.path.join(final_output_dir,
    #                                     'final_state.pth.tar')
    #     logger.info('=> loading model from {}'.format(model_state_file))
    #     model.load_state_dict(torch.load(model_state_file))

    # gpus = [int(i) for i in config.GPUS.split(',')]
    # model = torch.nn.DataParallel(model, device_ids=gpus).cuda()


    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        if(config.TEST.MODEL_FILE.split('/')[-1]=='checkpoint.pth.tar'):
            gpus = [int(i) for i in config.GPUS.split(',')]
            model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
            model.load_state_dict(torch.load(config.TEST.MODEL_FILE)['state_dict'])
        elif(config.TEST.MODEL_FILE.split('/')[-1]=='model_best.pth.tar'):  #multiGPU has model.module.
            gpus = [int(i) for i in config.GPUS.split(',')]
            model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
            model.state_dict(torch.load(config.TEST.MODEL_FILE))
        else:
            # model.load_state_dict(torch.load(config.TEST.MODEL_FILE)['model'], strict=False) #保存的no bn版本的权重导入
            if(int_adjust): #int推理
                model.load_state_dict(torch.load('output/weights_quan/int_mobilenetpose_nobn_refactor.pt')['model']) #保存的no bn版本的权重导入
            else:
                model.load_state_dict(torch.load(config.TEST.MODEL_FILE)['model']) #保存的no bn版本的权重导入
            # model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
            gpus = [int(i) for i in config.GPUS.split(',')]
            model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth.tar')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    ################################### int版本权重导入 ##########################################
    # int_model.load_state_dict(torch.load('output/weights_quan/int_mobilenetpose_nobn_refactor.pt')['model']) #保存的no bn版本的权重导入
    # model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    gpus = [int(i) for i in config.GPUS.split(',')]
    int_model = torch.nn.DataParallel(int_model, device_ids=gpus).cuda()
    #############################################################################################

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
            transforms.ToTensor(), #ToTensor()能够把灰度范围从0-255变换到0-1之间
            normalize, #进行归一化
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    # 浮点模型 evaluate on validation set
    validate(config, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir, int_adjust=int_adjust)

    # # int模型 evaluate on validation set
    # validate(config, valid_loader, valid_dataset, int_model, criterion,
    #          final_output_dir, tb_log_dir, int_adjust=True)

    print(len(models.pose_mobilenet_relu_bnfuse.fmap_block['input']))
    print(models.pose_mobilenet_relu_bnfuse.fmap_block['input'][0].shape)
    print(len(models.pose_mobilenet_relu_bnfuse.fmap_block['output']))
    print(models.pose_mobilenet_relu_bnfuse.fmap_block['output'][0].shape)


if __name__ == '__main__':
    main()
