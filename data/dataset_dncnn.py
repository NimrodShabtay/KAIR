import os.path

import cv2
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
from utils.dip_utils import get_noise

class DatasetDnCNN(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(DatasetDnCNN, self).__init__()
        print('Dataset: Denosing on AWGN with fixed sigma. Only dataroot_H is needed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 64
        self.sigma = opt['sigma'] if opt['sigma'] else 25
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else self.sigma

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)

        L_path = H_path

        if self.opt['phase'] == 'train':
            """
            # --------------------------------
            # get L/H patch pairs
            # --------------------------------
            """
            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = img_H.clone()

            # --------------------------------
            # add noise
            # --------------------------------
            noise = torch.randn(img_L.size()).mul_(self.sigma/255.0)
            img_L.add_(noise)

        else:
            """
            # --------------------------------
            # get L/H image pairs
            # --------------------------------
            """
            img_H = util.uint2single(img_H)
            img_L = np.copy(img_H)

            # --------------------------------
            # add noise
            # --------------------------------
            np.random.seed(seed=0)
            img_L += np.random.normal(0, self.sigma_test/255.0, img_L.shape)

            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            img_L = util.single2tensor3(img_L)
            img_H = util.single2tensor3(img_H)

        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'L_path': L_path}

    def __len__(self):
        return len(self.paths_H)


class DatasetDnCNNDIP(DatasetDnCNN):
    def __init__(self, opt):
        super(DatasetDnCNNDIP, self).__init__(opt)
        self.H_path = ['./trainsets/trainH/F16_GT.png',  './trainsets/trainH/kate.png'][0]
        self.GT = util.imread_uint(self.H_path, self.n_channels)
        # self.GT = cv2.resize(self.GT, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LANCZOS4)
        self.GT = util.uint2tensor3(self.GT).detach()
        self.net_input = get_noise(self.n_channels, 'noise', (self.GT.shape[1], self.GT.shape[2])).detach()
        self.noise = torch.randn(*self.GT.shape).mul_(self.sigma / 255.0).detach()

    def plot_images_before_start(self):
        image_name_ext = os.path.basename(self.H_path)
        img_name, ext = os.path.splitext(image_name_ext)
        img_dir = os.path.join(self.opt['path']['images'], img_name)
        util.mkdir(img_dir)

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        GT_path = self.H_path  # self.paths_H[index]
        # img_GT = util.imread_uint(GT_path, self.n_channels)

        L_path = GT_path

        if self.opt['phase'] == 'train':
            """
            # --------------------------------
            # get L/H patch pairs
            # --------------------------------
            """
            _, H, W = self.GT.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            # rnd_h = random.randint(0, max(0, H - self.patch_size))
            # rnd_w = random.randint(0, max(0, W - self.patch_size))
            #
            # img_GT = self.GT[:, rnd_h: rnd_h + self.patch_size, rnd_w: rnd_w + self.patch_size]
            # img_H = self.GT.clone().add_(self.noise).detach()[:, rnd_h: rnd_h + self.patch_size, rnd_w: rnd_w + self.patch_size]
            # img_L = self.net_input[:, rnd_h: rnd_h + self.patch_size, rnd_w: rnd_w + self.patch_size]

            # --------------------------------
            # For using resize
            # --------------------------------
            img_GT = self.GT
            noise = torch.randn(*self.GT.shape).mul_(self.sigma / 255.0).detach()
            img_H = img_GT.clone().add_(noise).detach()
            img_L = self.net_input

            # ------------------------------------------
            # Split the images to 4 patch sizes
            # ------------------------------------------
            # kernel_size, stride = self.patch_size, self.patch_size
            # patches = torch.torch.from_numpy(img_H).permute(2, 0, 1).unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
            # patches = patches.contiguous().view(patches.size(0), -1, kernel_size, kernel_size)
            # patch_H = patches[:, index].permute(1, 2, 0).numpy()
            #
            # noise_patches = net_input_H.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
            # noise_patches = noise_patches.contiguous().view(noise_patches.size(0), -1, kernel_size, kernel_size)
            # net_input = noise_patches[:, index]

        return {'L': img_L, 'H': img_H, 'GT': img_GT, 'H_path': GT_path, 'L_path': L_path}
