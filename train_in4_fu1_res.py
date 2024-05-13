#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2020/07/21 14:38:58
@Author  :   ZeyuXiao 
@Version :   1.0
@Contact :   zeyuxiao1997@163.com
@License :   (C)Copyright 2018-2019
@Desc    :   
'''
# here put the import lib

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import numpy as np
from models import Losses
from models import VHNet_in4_fu1_res
from utils.myutils import *
import torch.backends.cudnn as cudnn
from options import *
from datasets.datasetHYoutube import *
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
import random
import numpy as np
from tensorboardX import *
import torchvision.utils as visionutils
import math
# from thop import profile


def train():
    opt.ModelName = 'in4_fu1_res'
    print(opt)
    Best = 0
    transform = transforms.Compose([transforms.ToTensor()])
    opt.manualSeed = random.randint(1, 10000)
    opt.saveDir = os.path.join(opt.exp, opt.ModelName)
    create_exp_dir(opt.saveDir)
    device = torch.device("cuda:7")

    train_data = DatasetFromFolderTrainF7(opt)
    train_dataloader = DataLoader(train_data,
                        batch_size=opt.train_batch_size,
                        shuffle=True,
                        num_workers=opt.num_workers,
                        drop_last=True)
    print('length of train_dataloader: ',len(train_dataloader))
    last_epoch = 0

    ## initialize loss writer and logger
    ##############################################################
    loss_dir = os.path.join(opt.saveDir, 'loss')
    loss_writer = SummaryWriter(loss_dir)
    print("loss dir", loss_dir)
    trainLogger = open('%s/train.log' % opt.saveDir, 'w')
    ##############################################################

    model = VHNet_in4_fu1_res.VHNet(opt)
    model.train()
    model.cuda()
    # model = torch.nn.DataParallel(model)

    criterionCharb = Losses.CharbonnierLoss()
    criterionCharb.cuda()
    # criterionSSIM = LossNet.SSIMLoss()
    # criterionSSIM.cuda()


    lr = opt.lr
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=lr,
        betas=(opt.beta1, opt.beta2)
    )

    iteration = 0
    if opt.model_path != '':
        map_location = lambda storage, loc: storage
        checkpoint = torch.load(opt.model_path, map_location=map_location)
        last_epoch = checkpoint["epoch"]
        optimizer_state = checkpoint["optimizer"]
        optimizer.load_state_dict(optimizer_state)
        model.load_state_dict(checkpoint["model"])
        iteration = checkpoint["iteration"]
        lr = checkpoint["lr"]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('load pretrained')

    # if opt.model_path != '':
    #     print('=========> loading pretrained Teacher Network (EDVR)')
    #     map_location = lambda storage, loc: storage
    #     checkpoint = torch.load(opt.model_path, map_location=map_location)["model"]
    #     from collections import OrderedDict
    #     new_state_dict = OrderedDict()
    #     for k, v in checkpoint.items():
    #         # name = k[7:] # remove `module.`
    #         name = k
    #         new_state_dict[name] = v
    #     optimizer_state = new_state_dict
    #     model.load_state_dict(optimizer_state)
    #     last_epoch = checkpoint["epoch"]
    #     optimizer_state = checkpoint["optimizer"]
    #     optimizer.load_state_dict(optimizer_state)

    #     lr = checkpoint["lr"]
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    #     print('======> load pretrained Teacher Network (EDVR) successfully')
    # ##############################################################
    
    AllPSNR = 0
    for epoch in range(opt.max_epoch):
        if epoch < last_epoch:
            continue
        for _, batch in enumerate(train_dataloader, 0):
            iteration += 1  # 总共的iteration次数

            inputs, masks, target = batch
            inputs, masks, target = inputs.cuda(), masks.cuda(), target.cuda()
            print(inputs.shape, masks.shape, target.shape)
            out, out_FG = model(inputs, masks)
            print(out.shape)
            
            optimizer.zero_grad()
            targetFG = masks[:,3,:,:,:] * target
            CharbLoss = criterionCharb(out_FG, targetFG)
    #         # SSIMLoss = (1-criterionSSIM(out, target))/10 #数量级一致
            AllLoss = CharbLoss
            AllLoss.backward()
            optimizer.step()

            prediction_FG = torch.clamp(out_FG,0.0,1.0)
            prediction = torch.clamp(out,0.0,1.0)
            if iteration%2 == 0:
                PPsnr = compute_psnr(tensor2np(out_FG[0,:,:,:]),tensor2np(targetFG[0,:,:,:]))
                if PPsnr==float('inf'):
                    PPsnr=50
                AllPSNR += PPsnr
                print('[%d/%d][%d] AllLoss:%.10f|CharbLoss:%.10f|PSNR:%.6f'
                    % (epoch, opt.max_epoch, iteration,
                    AllLoss.item(),CharbLoss.item(), PPsnr))
                trainLogger.write(('[%d/%d][%d] AllLoss:%.10f|CharbLoss:%.10f|PSNR:%.6f'
                    % (epoch, opt.max_epoch, iteration,
                    AllLoss.item(),CharbLoss.item(), PPsnr))+'\n')

                # loss_writer.add_scalar('AllLoss', AllLoss.item(), iteration)
                loss_writer.add_scalar('CharbLoss', CharbLoss.item(), iteration)
                # loss_writer.add_scalar('SSIMLoss', SSIMLoss.item(), iteration)
                loss_writer.add_scalar('PSNR', PPsnr, iteration)
                trainLogger.flush()

            if iteration%1000 == 0:
                loss_writer.add_image('Prediction_FG', prediction_FG[0,:,:,:], iteration)
                loss_writer.add_image('Prediction', prediction[0,:,:,:], iteration)
                loss_writer.add_image('target', target[0,:,:,:], iteration)
                loss_writer.add_image('target_FG', targetFG[0,:,:,:], iteration)
                loss_writer.add_image('inputs', inputs[0,3,:,:,:], iteration)
                loss_writer.add_image('inputs_FG', masks[0,3,:,:,:] * inputs[0,3,:,:,:], iteration)

                
            if iteration % opt.saveStep == 0:
                is_best = AllPSNR > Best
                Best = max(AllPSNR, Best)
                if is_best or iteration%(opt.saveStep*3)==0:
                    prefix = opt.saveDir+'/VHNet_iter{}'.format(iteration)+'+PSNR'+str(Best)
                    file_name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
                    checkpoint = {
                        'epoch': epoch,
                        'iteration': iteration,
                        "optimizer": optimizer.state_dict(),
                        "model": model.state_dict(),
                        "lr": lr
                    }
                torch.save(checkpoint, file_name)
                print('model saved to ==>'+file_name)
                AllPSNR = 0

            if (iteration + 1) % opt.decay_step == 0:
                lr = lr * opt.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

    trainLogger.close()



if __name__ == "__main__":
    train()
