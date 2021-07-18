# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os

import numpy as np
import torch

from core.config import get_model_name
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images


logger = logging.getLogger(__name__)
diy_preprocess=True #True False


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #print("----------------------Train----------------------\n model:",model)
        # compute output
        output = model(input)
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        #print("output.shape=",output.shape)
        #print("target.shape=",target.shape)

        loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      #epoch, i, len(train_loader)//4, batch_time=batch_time,
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)
        # #Just for debug       reduce number of images to reduce training time
        # if(i==len(train_loader)//4):
        #     break

def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, int_adjust=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    #print("----------------------Validate----------------------\n model:",model)
    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3),
                         dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            input=input.type(torch.float32)
            #diy_preprocess==True   torch.Size([256, 256, 192, 3]) tensor(255.) tensor(0.) tensor(89.9854)    
            #diy_preprocess==False  torch.Size([256, 3, 256, 192]) tensor(2.6400) tensor(-2.1179) tensor(-0.4242) 
            # print(input.shape,torch.max(input), torch.min(input), torch.mean(input)) 
            # print(input)
            if(int_adjust==True):
                if(diy_preprocess==True): #需要自己进行[0,255]->[0,1]的浮点转换，以及-mean, /std的操作  RGB数据
                    BIT=16
                    M0=torch.tensor([54201, 55411, 55165]).reshape(1,-1,1,1) #在通道方向 [1, 3, 1, 1]
                    const0_16=torch.tensor([6703358, 6443221, 5711231]).reshape(1,-1,1,1)
                    round_15=torch.tensor([32768, 32768, 32768]).reshape(1,-1,1,1)
                    input=input.permute(0, 3, 1, 2).type(torch.float32)
                    # qfeaturemap0=input[0].detach().cpu().numpy().transpose(1,2,0).reshape(-1,input[0].shape[0]) #[3,256,192]->[256,192,3]->[256*192,3]
                    # qfeaturemap0.astype(np.int8).tofile('output/weights_quan_deconv3/input_preprocess/'+'input_256x192_0_255.bin')
                    # np.savetxt('output/weights_quan_deconv3/input_preprocess/'+'qinput0_0_255.txt', qfeaturemap0, fmt="%d", delimiter='  ') 
                    input = torch.tensor((input*M0 - const0_16).numpy()//65536).type(torch.int32).clamp_(-128,127).type(torch.float32) #从[-2.1179，2.64] 映射到 [-540,673]   + round_15
                    # input = ((input*M0 - const0_16)//65536).type(torch.int32).clamp_(-128,127).type(torch.float32) #从[-2.1179，2.64] 映射到 [-540,673]  
                    # python中直接变为整型是向0靠拢的，-32.8=-32，但硬件中移位附属补码保存，会直接变成-33
                    # input = ((input*M0 - const0_16 )>>16).clamp_(-128,127).type(torch.float32) #从[-2.1179，2.64] 映射到 [-540,673]

                    # print(input[0])
                    # data = ((data * M0_list.item()[Mkey].to(data.device) + 2**(BIT-1)).type(torch.int32)>>(BIT)).clamp_(-128, 127).type(torch.float32)#.type(torch.int32) # *M0并移位
                else: #diy_preprocess==False
                    input = torch.round(input *127.5/2.64).clamp_(-128,127) #从[-2.1179，2.64] 映射到 [-540,673]
                # print(input[0])
                # qfeaturemap0=input[0].detach().cpu().numpy().transpose(1,2,0).reshape(-1,input[0].shape[0]) #[3,256,192]->[256,192,3]->[256*192,3]
                # qfeaturemap0.astype(np.int8).tofile('output/weights_quan_deconv3/input_preprocess/'+'input_256x192.bin')
                # np.savetxt('output/weights_quan_deconv3/input_preprocess/'+'qinput0_afterpreprocess.txt', qfeaturemap0, fmt="%d", delimiter='  ') 
                # b=np.fromfile('output/weights_quan_deconv3/input_preprocess/'+'input_256x192.bin',dtype=np.int8)
                # # print(b.shape,input.shape) #input[0].shape torch.Size([3, 256, 192])
                # b=b.reshape(input[0].shape[1],input[0].shape[2],input[0].shape[0]).transpose(2,0,1).astype(np.float32) #(256, 192, 3)
                # # print(b.shape) #(3,256,192)
                # b=np.expand_dims(b,0) #[3,256,192]->[1,3,256,192]
                # input=torch.tensor(b).to(input.device)
            # print(input.shape,torch.max(input), torch.min(input), torch.mean(input.abs())) #torch.Size([128, 3, 256, 192]) tensor(127.) tensor(-102.) tensor(-21.4869)
            # for i in range(input.shape[0]):
            #     print(input[i].shape,torch.max(input[i]), torch.min(input[i]), torch.mean(input[i].abs())) #torch.Size([128, 3, 256, 192]) tensor(127.) tensor(-102.) tensor(-21.4869)

            # compute output
            output = model(input)
            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                output_flipped = model(input_flipped)
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]
                    # output_flipped[:, :, :, 0] = 0

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)
            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])
            if config.DATASET.DATASET == 'posetrack':
                filenames.extend(meta['filename'])
                imgnums.extend(meta['imgnum'].numpy())

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums)

        _, full_arch_name = get_model_name(config)
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, full_arch_name)
        else:
            _print_name_value(name_values, full_arch_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_acc', acc.avg, global_steps)
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars('valid', dict(name_value), global_steps)
            else:
                writer.add_scalars('valid', dict(name_values), global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
