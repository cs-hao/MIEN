import os
import os.path
import random
import math
import errno
import numpy as np
# import scipy.misc as misc
import imageio
import torch
import torch.utils.data as data
from torchvision import transforms

class MyImage(data.Dataset):
    def __init__(self, args, train=False):
        self.args = args
        self.train = False
        self.name = 'MyImage'
        self.scale = args.scale
        self.idx_scale = 0
        lr_apath = args.lrpath + '\\' + args.testset + '\\x' + str(args.scale[0])
        hr_apath = args.hrpath + '\\' + args.testset + '\\x' + str(args.scale[0])
        self.lr_filelist = []
        self.hr_filelist = []
        self.lrnamelist = []
        self.hrnamelist = []
        if not train:
            for f in os.listdir(lr_apath):
                try:
                    filename = os.path.join(lr_apath, f)
                    imageio.imread(filename)
                    self.lr_filelist.append(filename)
                    self.lrnamelist.append(f)
                except:
                    pass

            for f in os.listdir(hr_apath):
                try:
                    filename = os.path.join(hr_apath, f)
                    imageio.imread(filename)
                    self.hr_filelist.append(filename)
                    self.hrnamelist.append(f)
                except:
                    pass

    def __getitem__(self, idx):
        filename = os.path.split(self.lr_filelist[idx])[-1]
        filename, _ = os.path.splitext(filename)
        lr = misc.imread(self.lr_filelist[idx])
        lr = common.set_channel([lr], self.args.n_colors)[0]

        # filename = os.path.split(self.lr_filelist[idx])[-1]
        # filename, _ = os.path.splitext(filename)
        hr = misc.imread(self.hr_filelist[idx])
        hr = common.set_channel([hr], self.args.n_colors)[0]
        lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)
        return lr_tensor, hr_tensor, filename, self.idx_scale

    def __len__(self):
        return len(self.lr_filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

