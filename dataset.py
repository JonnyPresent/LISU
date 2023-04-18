"""
@FileName: dataset.py
@Time    : 4/30/2020
@Author  : Ning Zhang
@GitHub: https://github.com/noahzn
"""

from PIL import Image
from torch.utils.data import Dataset
import glob
from torchvision import transforms
import numpy as np
import torch
import cv2

import os


class LLRGBD_synthetic(Dataset):
    # two lighting
    def __init__(self, args, mode='train'):
        super().__init__()
        self.mode = mode
        self.base_folder = args.data_path
        self.to_tensor = transforms.ToTensor()

        if self.mode == 'train':
            rgb_folder = self.base_folder + 'selection_train2/*/rgb/*.jpg'
            rgb_folder2 = self.base_folder + 'selection_train_pair2/*/rgb/*.jpg'
            label_folder = self.base_folder + 'selection_train2/*/label/*.png'

        elif self.mode == 'val':
            rgb_folder = self.base_folder + 'selection_val2/*/rgb/*.jpg'
            rgb_folder2 = self.base_folder + 'selection_val_pair2/*/rgb/*.jpg'
            label_folder = self.base_folder + 'selection_val2/*/label/*.png'

        # if self.mode == 'train':
        #     rgb_folder = self.base_folder + 'selection_train2/11*/rgb/11*.jpg'
        #     rgb_folder2 = self.base_folder + 'selection_train_pair2/11*/rgb/11*.jpg'
        #     label_folder = self.base_folder + 'selection_train2/11*/label/11*.png'
        #
        # elif self.mode == 'val':
        #     rgb_folder = self.base_folder + 'selection_val2/11*/rgb/11*.jpg'
        #     rgb_folder2 = self.base_folder + 'selection_val_pair2/11*/rgb/11*.jpg'
        #     label_folder = self.base_folder + 'selection_val2/11*/label/11*.png'

        self.image_list = sorted(glob.glob(rgb_folder))
        self.image_list2 = sorted(glob.glob(rgb_folder2))
        self.label_list = sorted(glob.glob(label_folder))

        # self.image_list = self.image_list[:400]
        # self.image_list2 = self.image_list2[:400]
        # self.label_list = self.label_list[:400]
        # self.image_name = [x.split('/')[-1].split('.')[0] for x in self.image_list]

    def __getitem__(self, item):
        # name = self.image_name[item]

        # load image
        img = Image.open(self.image_list[item])
        img = img.resize((320, 240), Image.BILINEAR)  # w h c
        img = np.array(img)  # h w c

        img_dn = img
        img_dn = cv2.blur(img_dn, (5, 5))
        img_dn = torch.Tensor(img_dn).float().permute(2, 0, 1)  # c h w; torch.Size([3, 240, 320])

        img = Image.fromarray(img).convert('RGB')
        img = self.to_tensor(img)  # tensor c h w

        img2 = Image.open(self.image_list2[item])
        img2 = img2.resize((320, 240), Image.BILINEAR)
        img2 = np.array(img2)  # w h c

        img_dn2 = img2
        img_dn2 = cv2.blur(img_dn2, (2, 2))
        img_dn2 = torch.Tensor(img_dn2).float().permute(2, 0, 1)  # c h w

        img2 = Image.fromarray(img2).convert('RGB')
        img2 = self.to_tensor(img2)

        if self.mode == 'test':
            return img

        label = Image.open(self.label_list[item])
        label = label.resize((320, 240), Image.NEAREST)
        label = np.array(label)
        label = torch.from_numpy(label).long()

        return img, img2, label, img_dn, img_dn2

    def __len__(self):
        return len(self.image_list)


class LLRGBD_real(Dataset):
    # two lighting
    def __init__(self, args, mode='train'):
        super().__init__()
        self.mode = mode
        self.base_folder = args.data_path
        self.to_tensor = transforms.ToTensor()

        if self.mode == 'train':
            rgb_folder = self.base_folder + '/train_rgb_l/*.png'
            rgb_folder2 = self.base_folder + '/train_rgb_h/*.png'
            label_folder = self.base_folder + '/train_lbl/*.png'

        elif self.mode == 'val':
            rgb_folder = self.base_folder + '/val_rgb_l/*.png'
            rgb_folder2 = self.base_folder + '/val_rgb_h/*.png'
            label_folder = self.base_folder + '/val_lbl/*.png'

            # rgb_folder = self.base_folder + r'\val_rgb_l\*.png'
            # rgb_folder2 = self.base_folder + r'\val_rgb_h\*.png'
            # label_folder = self.base_folder + r'\val_lbl\*.png'

        self.image_list = sorted(glob.glob(rgb_folder))
        self.image_list2 = sorted(glob.glob(rgb_folder2))
        self.label_list = sorted(glob.glob(label_folder))

        self.image_name = [x.split('/')[-1].split('.')[0] for x in self.image_list]

    def __getitem__(self, item):
        name = self.image_name[item]

        # load image
        img = Image.open(self.image_list[item])
        img = img.resize((320, 240), Image.BILINEAR)
        img = np.array(img)

        img_dn = img
        img_dn = cv2.blur(img_dn, (3, 3))
        img_dn = img_dn * 1.0 / 255.0
        # Image.fromarray(img_dn).save('')

        img_dn = torch.Tensor(img_dn).float().permute(2, 0, 1)

        img = Image.fromarray(img).convert('RGB')
        img = self.to_tensor(img)
        # # ===
        img2 = Image.open(self.image_list2[item])
        img2 = img2.resize((320, 240), Image.BILINEAR)
        img2 = np.array(img2)

        img_dn2 = img2
        img_dn2 = cv2.blur(img_dn2, (5, 5))
        img_dn2 = img_dn2 * 1.0 / 255.0
        img_dn2 = torch.Tensor(img_dn2).float().permute(2, 0, 1)

        img2 = Image.fromarray(img2).convert('RGB')
        img2 = self.to_tensor(img2)

        if self.mode == 'test':
            return img, name

        label = Image.open(self.label_list[item])
        label = label.resize((320, 240), Image.NEAREST)
        label = np.array(label)
        label = torch.from_numpy(label).long()

        # #low-light, high, label
        # return img, img2, img_dn, label, name
        return img, img2, label, img_dn, img_dn2, name

    def __len__(self):
        return len(self.image_list)

