# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   createTxt4HYoutube.py
@Time    :   2021/12/22 22:09:07
@Author  :   ZeyuXiao
@Version :   1.0
@Contact :   zeyuxiao@mail.ustc.edu.cn, zeyuxiao1997@163.com
@License :   (C)Copyright 2019-2024
@Desc    :   None
"""

import os
import argparse


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def mainF7():
    # assert os.path.exists(inputdir), 'Input dir not found'
    # assert os.path.exists(maskdir), 'target dir not found'
    # assert os.path.exists(targetdir), 'target dir not found'

    groups = [line.rstrip() for line in open(os.path.join(txtdir))]
    # assert groups[0].startswith('/'), 'Paths from file_list must be absolute paths!'
    image_filevidss = [group.split(' ') for group in groups]
    # print(image_filevidss)
    mkdir(outputdir)
    for image in image_filevidss:
        real = image[0]
        mask = image[1]
        input = image[2]

        vids = sorted(os.listdir(os.path.join(rootdir,input)))
        # print(vids)
        leng = len(vids)
        for idx in range(0, leng):
            print(idx)
            groups = ''
            if idx == 0:
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+3]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            elif idx == 1:
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+3]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            elif idx == 2:
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+3]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            elif idx == (leng-3):
                groups += os.path.join(rootdir,input,vids[idx-3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            elif idx == (leng-2):
                groups += os.path.join(rootdir,input,vids[idx-3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            elif idx == (leng-1):
                groups += os.path.join(rootdir,input,vids[idx-3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            else:
                groups += os.path.join(rootdir,input,vids[idx-3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+3]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            print(groups)
            with open(os.path.join(outputdir, 'groupsTrainF7.txt'), 'a') as f:
                f.write(groups + '\n')


def mainF9():
    # assert os.path.exists(inputdir), 'Input dir not found'
    # assert os.path.exists(maskdir), 'target dir not found'
    # assert os.path.exists(targetdir), 'target dir not found'

    groups = [line.rstrip() for line in open(os.path.join(txtdir))]
    # assert groups[0].startswith('/'), 'Paths from file_list must be absolute paths!'
    image_filevidss = [group.split(' ') for group in groups]
    # print(image_filevidss)
    mkdir(outputdir)
    for image in image_filevidss:
        real = image[0]
        mask = image[1]
        input = image[2]

        vids = sorted(os.listdir(os.path.join(rootdir,input)))
        # print(vids)
        leng = len(vids)
        for idx in range(0, leng):
            print(idx)
            groups = ''
            if idx == 0:
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+4]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            elif idx == 1:
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+4]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            elif idx == 2:
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+4]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            elif idx == 3:
                groups += os.path.join(rootdir,input,vids[idx-3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+4]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            elif idx == (leng-4):
                groups += os.path.join(rootdir,input,vids[idx-4]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+3]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            elif idx == (leng-3):
                groups += os.path.join(rootdir,input,vids[idx-4]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            elif idx == (leng-2):
                groups += os.path.join(rootdir,input,vids[idx-4]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            elif idx == (leng-1):
                groups += os.path.join(rootdir,input,vids[idx-4]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            else:
                groups += os.path.join(rootdir,input,vids[idx-4]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+4]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            print(groups)
            with open(os.path.join(outputdir, 'groupsTrainF9.txt'), 'a') as f:
                f.write(groups + '\n')


def mainF11():
    # assert os.path.exists(inputdir), 'Input dir not found'
    # assert os.path.exists(maskdir), 'target dir not found'
    # assert os.path.exists(targetdir), 'target dir not found'

    groups = [line.rstrip() for line in open(os.path.join(txtdir))]
    # assert groups[0].startswith('/'), 'Paths from file_list must be absolute paths!'
    image_filevidss = [group.split(' ') for group in groups]
    # print(image_filevidss)
    mkdir(outputdir)
    for image in image_filevidss:
        real = image[0]
        mask = image[1]
        input = image[2]

        vids = sorted(os.listdir(os.path.join(rootdir,input)))
        leng = len(vids)

        for idx in range(0, leng):
            print(idx)
            groups = ''
            if idx == 0:
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+4]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+5]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+5].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            elif idx == 1:
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+4]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+5]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+5].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            elif idx == 2:
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+4]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+5]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+5].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            elif idx == 3:
                groups += os.path.join(rootdir,input,vids[idx-3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+4]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+5]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+5].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            elif idx == 4:
                groups += os.path.join(rootdir,input,vids[idx-4]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-4]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+4]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+5]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+5].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            elif idx == (leng-5):
                groups += os.path.join(rootdir,input,vids[idx-5]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-4]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+4]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+4]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-5].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            elif idx == (leng-4):
                groups += os.path.join(rootdir,input,vids[idx-5]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-4]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+3]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-5].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            elif idx == (leng-3):
                groups += os.path.join(rootdir,input,vids[idx-5]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-4]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-5].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            elif idx == (leng-2):
                groups += os.path.join(rootdir,input,vids[idx-5]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-4]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-5].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            elif idx == (leng-1):
                groups += os.path.join(rootdir,input,vids[idx-5]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-4]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-5].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            else:
                groups += os.path.join(rootdir,input,vids[idx-5]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-4]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+3]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+4]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+5]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-5].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+3].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+4].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+5].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            print(groups)
            with open(os.path.join(outputdir, 'groupsTrainF11.txt'), 'a') as f:
                f.write(groups + '\n')


def mainF5():
    # assert os.path.exists(inputdir), 'Input dir not found'
    # assert os.path.exists(maskdir), 'target dir not found'
    # assert os.path.exists(targetdir), 'target dir not found'

    groups = [line.rstrip() for line in open(os.path.join(txtdir))]
    # assert groups[0].startswith('/'), 'Paths from file_list must be absolute paths!'
    image_filevidss = [group.split(' ') for group in groups]
    # print(image_filevidss)
    mkdir(outputdir)
    for image in image_filevidss:
        real = image[0]
        mask = image[1]
        input = image[2]

        vids = sorted(os.listdir(os.path.join(rootdir,input)))
        leng = len(vids)

        for idx in range(0, leng):
            print(idx)
            groups = ''
            if idx == 0:
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            elif idx == 1:
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            elif idx == (leng-2):
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            elif idx == (leng-1):
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            else:
                groups += os.path.join(rootdir,input,vids[idx-2]) + '|'
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+2]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+2].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            print(groups)
            with open(os.path.join(outputdir, 'groupsTrainF5.txt'), 'a') as f:
                f.write(groups + '\n')


def mainF3():
    # assert os.path.exists(inputdir), 'Input dir not found'
    # assert os.path.exists(maskdir), 'target dir not found'
    # assert os.path.exists(targetdir), 'target dir not found'

    groups = [line.rstrip() for line in open(os.path.join(txtdir))]
    # assert groups[0].startswith('/'), 'Paths from file_list must be absolute paths!'
    image_filevidss = [group.split(' ') for group in groups]
    # print(image_filevidss)
    mkdir(outputdir)
    for image in image_filevidss:
        real = image[0]
        mask = image[1]
        input = image[2]

        vids = sorted(os.listdir(os.path.join(rootdir,input)))
        # print(vids)
        leng = len(vids)
        for idx in range(0, leng):
            print(idx)
            groups = ''
            if idx == 0:
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            elif idx == (leng-1):
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            else:
                groups += os.path.join(rootdir,input,vids[idx-1]) + '|'
                groups += os.path.join(rootdir,input,vids[idx]) + '|'
                groups += os.path.join(rootdir,input,vids[idx+1]) + '|'
                groups += os.path.join(rootdir,mask,vids[idx-1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,mask,vids[idx+1].replace('.jpg','.png')) + '|'
                groups += os.path.join(rootdir,real,vids[idx].replace('.png','.jpg'))
            print(groups)
            with open(os.path.join(outputdir, 'groupsTrainF3.txt'), 'a') as f:
                f.write(groups + '\n')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/gdata/xiaozy/HYouTube', metavar='PATH', help='root dir to save low resolution images')
    parser.add_argument('--txt', type=str, default='/gdata/xiaozy/HYouTube/train_list.txt', metavar='PATH', help='root dir to save low resolution images')
    parser.add_argument('--input', type=str, default='/gdata/xiaozy/HYouTube/synthetic_composite_videos', metavar='PATH', help='root dir to save low resolution images')
    parser.add_argument('--mask', type=str, default='/gdata/xiaozy/HYouTube/foreground_mask', metavar='PATH', help='root dir to save low resolution images')
    parser.add_argument('--target', type=str, default='/gdata/xiaozy/HYouTube/real_videos', metavar='PATH', help='root dir to save high resolution images')
    parser.add_argument('--output', type=str, default='/gdata/xiaozy/HYouTube/Groups', metavar='PATH', help='output dir to save group txt files')
    parser.add_argument('--ext', type=str, default='.jpg', help='Extension of files')
    args = parser.parse_args()

    txtdir = args.txt
    rootdir = args.root
    # inputdir = args.input
    # maskdir = args.mask
    # targetdir = args.target
    outputdir = args.output
    ext = args.ext

    mainF3()
    mainF5()
    mainF7()
    mainF9()
    mainF11()
    