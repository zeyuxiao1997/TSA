from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from torchvision.transforms import Compose, ToTensor
import numpy as np
import random
from models import Losses
from models import VHNet_in4_fu1_res
from utils.myutils import *
import torch.backends.cudnn as cudnn
from options import *
from loss import *
# from datasets import Dataset4Sintel_condition1_Baselines
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
from tensorboardX import *
import torchvision.utils as visionutils
import torchvision.transforms.functional as tf
from myutils.utils import *


def transform():
    return Compose([
        ToTensor(),
    ])

def get_image(img):
    img = Image.open(img).convert('RGB')
    return img

def inference():
    modelname = 'in4_fu1_res'
    model_dir = 'models'   # pth under this folder
    group_file = 'groupsTestF7.txt'
    save_root = '/gdata/xiaozy/Results4VideoHarmPSNR'
    save_root = os.path.join(save_root, modelname)

    groups = [line.rstrip() for line in open(group_file)]
    image_filenames = [group.split('|') for group in groups]
    length = len(image_filenames)
    # print(image_filenames)
    
    compute_lpips_all = perceptual_loss(net='alex')
    compute_ssim_all = ssim_loss()
    compute_psnr_all = psnr_loss()


    model = VHNet_in4_fu1_res.VHNet(opt)
    model.eval()
    model.cuda()
    
    pth = sorted(os.listdir(model_dir))
    # pth.sort(key= lambda x:int(x[11:11+4]))

    print(pth)

    for index in pth:
        if index.endswith('pth'):
            # 
            model_path = os.path.join(model_dir, index)
            map_location = lambda storage, loc: storage
            checkpoint = torch.load(model_path, map_location=map_location)
            model.load_state_dict(checkpoint["model"])
            print(model_path)
            print('load pretrained')

            savepath = os.path.join(save_root, index[:20])
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            logger = Logger_yaml(os.path.join(savepath, 'inference.yml'))
            metric_track = MetricTracker(['ssim', 'psnr', 'lpips'])
            metric_track.reset()
    


            for im in range(0,length,10):
                # print(image)
                images = [get_image(img) for img in image_filenames[im]]
                inputs = images[0:7]
                masks = images[7:14]
                target = images[-1]
                masks = [i.convert('1') for i in masks]

                inputs = [tf.resize(item, [256, 256]) for item in inputs]
                masks = [tf.resize(item, [256, 256]) for item in masks]
                target = tf.resize(target, [256, 256])

                target = transform()(target)
                inputs = [transform()(i) for i in inputs]
                masks = [transform()(j) for j in masks]

                inputs = torch.cat((torch.unsqueeze(inputs[0], 0), torch.unsqueeze(inputs[1], 0),
                                    torch.unsqueeze(inputs[2], 0), torch.unsqueeze(inputs[3], 0),
                                    torch.unsqueeze(inputs[4], 0), torch.unsqueeze(inputs[5], 0),
                                    torch.unsqueeze(inputs[6], 0)))

                masks = torch.cat((torch.unsqueeze(masks[0], 0), torch.unsqueeze(masks[1], 0),
                                    torch.unsqueeze(masks[2], 0), torch.unsqueeze(masks[3], 0),
                                    torch.unsqueeze(masks[4], 0), torch.unsqueeze(masks[5], 0),
                                    torch.unsqueeze(masks[6], 0)))

                inputs = torch.unsqueeze(inputs, 0)
                masks = torch.unsqueeze(masks, 0)
                target = torch.unsqueeze(target, 0)
            
                with torch.no_grad():
                    out, out_FG = model(inputs.cuda(), masks.cuda())
                    out = torch.clamp(out,0.0,1.0)
                    out_FG = torch.clamp(out_FG,0.0,1.0)
           
                targetFG = masks[:,3,:,:,:] * target
                psnr1 = compute_psnr_all(out_FG[0].cpu(),targetFG[0].cpu()).item()
                ssim1 = compute_ssim_all(tensor2np(out_FG[0].cpu()),tensor2np(targetFG[0].cpu())).item()
                lpips1 = compute_lpips_all(out_FG[0].cpu(),targetFG[0].cpu()).item()
                
                metric_track.update('psnr', psnr1)
                metric_track.update('lpips', lpips1)
                metric_track.update('ssim', ssim1)
                
            result = metric_track.result()
            all_data = metric_track.all_data()
            logger.log_dict(result, 'evaluation results')
            logger.log_dict(all_data, 'all data')
    
        

if __name__ == "__main__":
    inference()
