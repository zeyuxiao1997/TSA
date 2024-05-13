# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   datasetHYoutube.py
@Time    :   2021/11/26 15:01:44
@Author  :   ZeyuXiao
@Version :   1.0
@Contact :   zeyuxiao@mail.ustc.edu.cn, zeyuxiao1997@163.com
@License :   (C)Copyright 2019-2024
@Desc    :   None
"""

import torch.utils.data as data
import torch
from torchvision.transforms import Compose, ToTensor
import os
import random
from PIL import Image,ImageOps
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import cv2
import numpy as np
import torchvision.transforms.functional as tf


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def get_patch(img_nn, img_tar, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_nn[0].size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)


    img_tar = img_tar.crop((iy, ix, iy + ip, ix + ip))  # [:, ty:ty + tp, tx:tx + tp]
    img_nn = [j.crop((iy, ix, iy + ip, ix + ip)) for j in img_nn]  # [:, iy:iy + ip, ix:ix + ip]

    return  img_nn, img_tar


def augment(inputs, masks, target, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        target = ImageOps.flip(target)
        inputs = [ImageOps.flip(j) for j in inputs]
        masks = [ImageOps.flip(j) for j in masks]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            target = ImageOps.mirror(target)
            inputs = [ImageOps.mirror(j) for j in inputs]
            masks = [ImageOps.mirror(j) for j in masks]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            target = target.rotate(180)
            inputs = [j.rotate(180) for j in inputs]
            masks = [j.rotate(180) for j in masks]
            info_aug['trans'] = True

    return  inputs, masks, target

def get_image(img):
    img = Image.open(img).convert('RGB')
    return img


def load_image_trainF7(group):
    images = [get_image(img) for img in group]
    inputs = images[0:7]
    masks = images[7:14]
    target = images[-1]
    masks = [i.convert('1') for i in masks]
    return inputs, masks, target


def transform():
    return Compose([
        ToTensor(),
    ])


class DatasetFromFolderTrainF7(data.Dataset):
    """
    For test dataset, specify
    `group_file` parameter to target TXT file
    data_augmentation = None
    black_edge_crop = None
    flip = None
    rot = None
    """
    def __init__(self, opt, transform=transform()):
        super(DatasetFromFolderTrainF7, self).__init__()
        groups = [line.rstrip() for line in open(os.path.join(opt.group_file))]
        self.image_filenames = [group.split('|') for group in groups]
        self.transform = transform
        self.data_augmentation = opt.augmentation
        self.hflip = opt.hflip
        self.rot = opt.rot
        self.resize = opt.resize

    def __getitem__(self, index):
        inputs, masks, target = load_image_trainF7(self.image_filenames[index])
        
        # 1280x720-- 270x480
        inputs = [tf.resize(item, [self.resize, self.resize]) for item in inputs]
        masks = [tf.resize(item, [self.resize, self.resize]) for item in masks]
        target = tf.resize(target, [self.resize, self.resize])

        if self.data_augmentation:
            inputs, masks, target = augment(inputs, masks, target, self.hflip, self.rot)

        if self.transform:
            target = self.transform(target)
            inputs = [self.transform(i) for i in inputs]
            masks = [self.transform(j) for j in masks]

        inputs = torch.cat((torch.unsqueeze(inputs[0], 0), torch.unsqueeze(inputs[1], 0),
                             torch.unsqueeze(inputs[2], 0), torch.unsqueeze(inputs[3], 0),
                             torch.unsqueeze(inputs[4], 0), torch.unsqueeze(inputs[5], 0),
                             torch.unsqueeze(inputs[6], 0)))

        masks = torch.cat((torch.unsqueeze(masks[0], 0), torch.unsqueeze(masks[1], 0),
                             torch.unsqueeze(masks[2], 0), torch.unsqueeze(masks[3], 0),
                             torch.unsqueeze(masks[4], 0), torch.unsqueeze(masks[5], 0),
                             torch.unsqueeze(masks[6], 0)))

        return inputs, masks, target

    def __len__(self):
        return len(self.image_filenames)


class DatasetFromFolderTestF7(data.Dataset):
    """
    For test dataset, specify
    `group_file` parameter to target TXT file
    data_augmentation = None
    black_edge_crop = None
    flip = None
    rot = None
    """
    def __init__(self, opt, transform=transform()):
        super(DatasetFromFolderTestF7, self).__init__()
        groups = [line.rstrip() for line in open(os.path.join(opt.group_file))]
        self.image_filenames = [group.split('|') for group in groups]
        self.transform = transform
        self.data_augmentation = opt.augmentation
        self.patch_size = opt.patch_size
        self.hflip = opt.hflip
        self.rot = opt.rot

    def __getitem__(self, index):
        inputs, masks, target = load_image_trainF7(self.image_filenames[index])
        
        # 1280x720-- 270x480
        inputs = [tf.resize(item, [270, 480]) for item in inputs]
        masks = [tf.resize(item, [270, 480]) for item in masks]
        target = tf.resize(target, [270, 480])

        if self.transform:
            target = self.transform(target)
            inputs = [self.transform(i) for i in inputs]
            masks = [self.transform(j) for j in masks]

        inputs = torch.cat((torch.unsqueeze(inputs[0], 0), torch.unsqueeze(inputs[1], 0),
                             torch.unsqueeze(inputs[2], 0), torch.unsqueeze(inputs[3], 0),
                             torch.unsqueeze(inputs[4], 0), torch.unsqueeze(inputs[5], 0),
                             torch.unsqueeze(inputs[6], 0)))

        masks = torch.cat((torch.unsqueeze(masks[0], 0), torch.unsqueeze(masks[1], 0),
                             torch.unsqueeze(masks[2], 0), torch.unsqueeze(masks[3], 0),
                             torch.unsqueeze(masks[4], 0), torch.unsqueeze(masks[5], 0),
                             torch.unsqueeze(masks[6], 0)))

        return inputs, masks, target

    def __len__(self):
        return len(self.image_filenames)


if __name__ == '__main__':
    output = 'visualize'
    if not os.path.exists(output):
        os.mkdir(output)
    dataset = DatasetFromFolder(4, True, 'dataset/groups.txt', 64, True, True, True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=False)
    for i, (inputs, target) in enumerate(dataloader):
        if i > 10:
            break
        if not os.path.exists(os.path.join(output, 'group{}'.format(i))):
            os.mkdir(os.path.join(output, 'group{}'.format(i)))
        input0, input1, input2, input3, input4 = inputs[0][0], inputs[0][1], inputs[0][2], inputs[0][3], inputs[0][4]
        vutils.save_image(input0, os.path.join(output, 'group{}'.format(i), 'input0.png'))
        vutils.save_image(input1, os.path.join(output, 'group{}'.format(i), 'input1.png'))
        vutils.save_image(input2, os.path.join(output, 'group{}'.format(i), 'input2.png'))
        vutils.save_image(input3, os.path.join(output, 'group{}'.format(i), 'input3.png'))
        vutils.save_image(input4, os.path.join(output, 'group{}'.format(i), 'input4.png'))
        vutils.save_image(target, os.path.join(output, 'group{}'.format(i), 'target.png'))