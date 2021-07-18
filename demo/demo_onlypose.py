from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import shutil

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
import time


import _init_paths
import models
from core.config import config
from core.config import update_config
from core.function import get_final_preds
from core.inference import get_max_preds
from utils.transforms import get_affine_transform

cfg=config

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

SKELETON = [
    [1,3],[1,0],[2,4],[2,0],[0,5],[0,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]
]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NUM_KPTS = 17

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def draw_pose(keypoints,img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert keypoints.shape == (NUM_KPTS,2)
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0],keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0],keypoints[kpt_b][1] 
        cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)

def draw_bbox(box,img):
    """draw the detected bounding box on the image.
    :param img:
    """
    cv2.rectangle(img, box[0], box[1], color=(0, 255, 0),thickness=3)


def get_pose_estimation_prediction(pose_model, image): #, center, scale
    rotation = 0

    # # pose estimation transformation
    # trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE) #trans是仿射变换矩阵 cfg.MODEL.IMAGE_SIZE=192,256
    # model_input = cv2.warpAffine(
    #     image,
    #     trans,
    #     (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
    #     flags=cv2.INTER_LINEAR)

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225]),
    # ])
    # # print(image.shape) #(256, 192, 3)
    # # pose estimation inference
    # model_input = transform(image).unsqueeze(0)
    # # print(model_input.shape) #torch.Size([1, 3, 256, 192])

    diy_preprocess=True
    if(diy_preprocess==True): #需要自己进行[0,255]->[0,1]的浮点转换，以及-mean, /std的操作  RGB数据
        BIT=16
        M0=torch.tensor([54201, 55411, 55165]).reshape(1,-1,1,1) #在通道方向 [1, 3, 1, 1]
        const0_16=torch.tensor([6703358, 6443221, 5711231]).reshape(1,-1,1,1)
        input=torch.tensor(image).unsqueeze(0).permute(0, 3, 1, 2).type(torch.float32)
        # qfeaturemap0=input[0].detach().cpu().numpy().transpose(1,2,0).reshape(-1,input[0].shape[0]) #[3,256,192]->[256,192,3]->[256*192,3]
        # qfeaturemap0.astype(np.int8).tofile('output/weights_quan_deconv3/input_preprocess/'+'input_256x192_0_255.bin')
        # np.savetxt('output/weights_quan_deconv3/input_preprocess/'+'qinput0_0_255.txt', qfeaturemap0, fmt="%d", delimiter='  ') 
        model_input = torch.tensor((input*M0 - const0_16).numpy()//65536).type(torch.int32).clamp_(-128,127).type(torch.float32) #从[-2.1179，2.64] 映射到 [-540,673]   + round_15
        # input = ((input*M0 - const0_16)//65536).type(torch.int32).clamp_(-128,127).type(torch.float32) #从[-2.1179，2.64] 映射到 [-540,673]  
        # python中直接变为整型是向0靠拢的，-32.8=-32，但硬件中移位附属补码保存，会直接变成-33
        # input = ((input*M0 - const0_16 )>>16).clamp_(-128,127).type(torch.float32) #从[-2.1179，2.64] 映射到 [-540,673]

        # print(input[0])
        # data = ((data * M0_list.item()[Mkey].to(data.device) + 2**(BIT-1)).type(torch.int32)>>(BIT)).clamp_(-128, 127).type(torch.float32)#.type(torch.int32) # *M0并移位
    else: #diy_preprocess==False
        data = image.transpose(2,0,1)  #[h,w,c]->[c,h,w]
        # data = resize_img.transpose(2,0,1)  #[h,w,c]->[c,h,w]
        # data = rgb_img
        # data=np.array(np.random.randint(low=0,high=255,size=(3,256,192)))
        data = data.reshape((1,3,256, 192)).astype(np.float32)/255.  #变成四维的
        data = np.array(data) #[0,1]
        mean = np.array([0.485, 0.456, 0.406]).reshape(1,-1,1,1) 
        std  = np.array([0.229, 0.224, 0.225]).reshape(1,-1,1,1)
        model_input = torch.tensor((data-mean)/std).type(torch.float32) #torch.float64->torch.float32

        model_input = torch.round(model_input *127.5/2.64).clamp_(-128,127) #从[-2.1179，2.64] 映射到 [-540,673]

    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)
        # preds, _ = get_final_preds(
        #     cfg,
        #     output.clone().cpu().numpy(),
        #     np.asarray([center]),
        #     np.asarray([scale]))

        coords, maxvals = get_max_preds(output.clone().cpu().numpy())
        preds = coords.copy()
        return preds, maxvals



def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default='experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3_mobile.yaml') #demo/inference-config-simplebaselines-res50.yaml
    parser.add_argument('--video', type=str)#, default='demo/video/dance_30.mp4') #experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3_mobile.yaml
    parser.add_argument('--webcam',action='store_true')
    parser.add_argument('--image',type=str)#, default='demo/test_val2017/3.jpg')
    parser.add_argument('--image_dir',type=str, default='demo/nonliving/')#, default='demo/test_val2017/') test nonliving
    parser.add_argument('--write',action='store_false') #'store_true'
    parser.add_argument('--showFps',action='store_true')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase  
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def main():
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    # update_config(cfg, args)
    update_config(args.cfg)

    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    box_model.to(CTX)
    box_model.eval()

    ###################################################################################
    int_adjust=True    #在这儿修改float 还是int型
    ###################################################################################

    # for shufflenetv2
    shufflenetv2_spec = {'0.5': ([4, 8, 4], [24, 48, 96, 192, 1024]),
                         '1.0': ([4, 8, 4], [24, 116, 232, 464, 1024]),
                         '1.5': ([4, 8, 4], [24, 176, 352, 704, 1024]),
                         '2.0': ([4, 8, 4], [24, 244, 488, 976, 2048])}
    stages_repeats, stages_out_channels = shufflenetv2_spec['1.0']
    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, 
        stages_repeats, stages_out_channels,
        is_train=False,
        int_adjust=int_adjust, #False是float版本（默认），True是int版本
    )

    if cfg.TEST.MODEL_FILE:
        if(int_adjust):
            int_modelfile='output/weights_quan_deconv3/int_mobilenetpose_shortcut0.pt' #_deconv3
            pose_model.load_state_dict(torch.load(int_modelfile), strict=True) #保存的no bn版本的权重导入
            print('*****Finally load model from',int_modelfile,'*****')
        else:
            print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
            pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE)['model'], strict=True) #False
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    gpus = [int(i) for i in config.GPUS.split(',')] 
    pose_model = torch.nn.DataParallel(pose_model, device_ids=gpus)
    pose_model.to(CTX)
    pose_model.eval()

    # Loading an video or an image or webcam 
    if args.webcam:
        vidcap = cv2.VideoCapture(0)
    elif args.video:
        vidcap = cv2.VideoCapture(args.video)
    elif args.image:
        image_bgr = cv2.imread(args.image)
    elif args.image_dir:
        image_dir = args.image_dir
    else:
        print('please use --video or --webcam or --image or --image_dir to define the input.')
        return 

    if args.webcam or args.video:
        if args.write:
            save_path = 'demo/video/dance_30_output.avi'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(save_path,fourcc, 24.0, (int(vidcap.get(3)),int(vidcap.get(4))))
        while True:
            ret, image_bgr = vidcap.read()
            if ret:
                last_time = time.time()
                image = image_bgr[:, :, [2, 1, 0]]

                input = []
                img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img/255.).permute(2,0,1).float().to(CTX)
                input.append(img_tensor)

                # object detection box
                pred_boxes = get_person_detection_boxes(box_model, input, threshold=0.9)

                # pose estimation
                if len(pred_boxes) >= 1:
                    for box in pred_boxes:
                        center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                        image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
                        pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
                        if len(pose_preds)>=1:
                            for kpt in pose_preds:
                                draw_pose(kpt,image_bgr) # draw the poses

                if args.showFps:
                    fps = 1/(time.time()-last_time)
                    img = cv2.putText(image_bgr, 'fps: '+ "%.2f"%(fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

                if args.write:
                    out.write(image_bgr)

                cv2.imshow('demo',image_bgr)
                if cv2.waitKey(1) & 0XFF==ord('q'):
                    break
            else:
                print('cannot load the video.')
                break

        cv2.destroyAllWindows()
        vidcap.release()
        if args.write:
            print('video has been saved as {}'.format(save_path))
            out.release()

    else: # estimate on the image
        if(args.image):
            index=1
        else: #args.image_dir
            index=8

        count=0
        for i in range(index):
            if(args.image_dir):
                img_path=image_dir+str(i)+'.jpg'
                image_bgr = cv2.imread(img_path) #opencv默认读取是 BGR 3通道，numpy数组，uint8[0-255]

            last_time = time.time()
            image = image_bgr[:, :, [2, 1, 0]]

            input = []
            img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img/255.).permute(2,0,1).float().to(CTX)
            # print(img_tensor.shape) #torch.Size([3, 256, 192])
            input.append(img_tensor)
            # print(len(input),type(input)) #1 <class 'list'>

            image_pose = image_bgr.copy()
            pose_preds,maxvals = get_pose_estimation_prediction(pose_model, image_pose) #pose_preds[1,17,2]  maxvals[1,17,1]
            point_index = []
            if len(pose_preds)>=1:
                # print("###########################################################\n",pose_preds*4,"###########################################################\n")
                for num,kpt in enumerate(pose_preds[0]):
                    # assert kpt.shape == (NUM_KPTS,2)
                    # draw_pose(kpt*4,image_bgr) # draw the poses
                    # print(maxvals[0][num])
                    if(int_adjust==True):
                        MThre=[5579,4255,4563,3438,3726,1647,5097,4778,5012,5419,4625,4787,5777,4312,4487,2317,2932] #阈值为0.2时
                        if np.max(maxvals[0][num]) > MThre[num]: #去掉置信度较小的关节点
                            count+=1
                            point_index.append(num)
                            print("### 0 ### This is",i,"image, and The threshold of its",num,"keypoint is",np.max(maxvals[0][num]))
                        else:
                            print("### 1 ### This is",i,"image, and its",num,"keypoints are under Threshold. The threshold is",np.max(maxvals[0][num]),'/',MThre[num])
                    else: #int_adjust=False
                        if np.max(maxvals[0][num]) > 0.2: #去掉置信度较小的关节点
                            count+=1
                            point_index.append(num)
                            print("### 0 ### This is",i,"image, and The threshold of its",num,"keypoint is",np.max(maxvals[0][num]))
                        else:
                            print("### 1 ### This is",i,"image, and its",num,"keypoints are under Threshold. The threshold is",np.max(maxvals[0][num]))
                keypoints=pose_preds[0]*4
                print('point_index:',point_index)
                # print('keypoints:',keypoints)
                    
                for j in point_index:
                    x_a, y_a = keypoints[j][0],keypoints[j][1]
                    cv2.circle(image_bgr, (int(x_a), int(y_a)), 4, CocoColors[j], -1)
                    # cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
                    # cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)

            # # object detection box
            # pred_boxes = get_person_detection_boxes(box_model, input, threshold=0.9)
            # print(len(pred_boxes),type(pred_boxes),pred_boxes[0],pred_boxes[1],pred_boxes[2]) #3 <class 'list'> [(8.489238, 39.513348), (158.91432, 239.99818)] [(28.99751, 32.693623), (67.57304, 128.24908)] [(56.58733, 39.213516), (73.77373, 101.664635)]

            # # pose estimation
            # if len(pred_boxes) >= 1:
            #     for box in pred_boxes:
            #         center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
            #         # image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
            #         image_pose = image_bgr.copy()
            #         pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
            #         if len(pose_preds)>=1:
            #             for kpt in pose_preds:
            #                 draw_pose(kpt,image_bgr) # draw the poses
            
            if args.showFps:
                fps = 1/(time.time()-last_time)
                img = cv2.putText(image_bgr, 'fps: '+ "%.2f"%(fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            if args.write:
                save_path = 'demo/mobilenetv2/test_nonliving_image'+str(i)+'.jpg'
                cv2.imwrite(save_path,image_bgr)
                print('the result image has been saved as {}'.format(save_path))

            # cv2.imshow('demo',image_bgr)
            # if cv2.waitKey(0) & 0XFF==ord('q'):
            #     cv2.destroyAllWindows()
        print('count=',count)
        
if __name__ == '__main__':
    main()
