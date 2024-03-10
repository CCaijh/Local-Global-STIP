from __future__ import print_function, division

import os.path

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import codecs
from core.utils import preprocess
from PIL import Image
import logging

class Norm(object):
    def __init__(self, max=255):
        self.max = max

    def __call__(self, sample):
        video_x = sample
        new_video_x = video_x / self.max
        return new_video_x


class ToTensor(object):

    def __call__(self, sample):
        video_x = sample
        video_x = video_x.transpose((0, 3, 1, 2))
        video_x = np.array(video_x)
        return torch.from_numpy(video_x).float()


class sf(Dataset):

    def __init__(self, configs, seqences, mode, transform=None):
        self.transform = transform
        self.mode = mode
        self.configs = configs
        self.patch_size = configs.patch_size
        self.img_width = configs.img_width
        self.img_height = configs.img_height
        self.img_channel = configs.img_channel

        logger = logging.getLogger("stip.train")
        logger.info('Loading {} dataset'.format(self.mode))
        self.file_list = seqences
        logger.info('Loading {} dataset finished, with size {}:'.format(self.mode, len(self.file_list)))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # print(self.index[3])
        sequence = self.file_list[idx]
        assert len(sequence) == 10

        data_slice = np.ndarray(shape=(10, self.img_height, self.img_width, self.img_channel), dtype=np.uint8)
        for i in range(10):
            image = Image.open(os.path.join(self.configs.image_path,sequence[i]))
            image = image.resize((self.img_width,self.img_width))
            try:
                data_slice[i, :] = np.array(image)
            except:
                print(data_slice.shape)
                print('%d,%s'%(i,os.path.join(self.configs.image_path,sequence[i])))

        #print("origin input image sequence : ", data_slice.shape)  #10,512,512,3 = 10 512 * 512 images with rgb
        video_x = preprocess.reshape_patch(data_slice, self.patch_size)
        #print("reshape_patch :", video_x.shape) # 10,64,64,192
        sample = video_x

        if self.transform:
            sample = self.transform(sample)

        return sample