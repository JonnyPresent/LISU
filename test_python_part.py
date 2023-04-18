

import csv

import cv2
import numpy as np

# with open('log/log.csv', 'a') as csvfile:
#     wt = csv.DictWriter(csvfile, fieldnames=['acc', 'macc', 'miou', 'average'])
#
#     wt.writeheader()
#     wt.writerow({'acc': 1, 'macc': 2, 'miou': 4, 'average': (1 + 3) / 2})
# batch_size = 4
# fog_factor_low = [0] * batch_size
# (for i in range(16))
import torch
from PIL.Image import Image
from torchvision import transforms

# def gen_mask(self, image, img_nf):
#     # img_nf = image.clone().permute(0, 2, 3, 1).numpy()
#     # img_nf = cv2.blur(img_nf, (5, 5))
#     # img_nf = torch.Tensor(img_nf).float().permute(2, 0, 1)
#
#     dark = image
#     # dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
#     dark = dark[0:1, :, :] * 0.299 + dark[1:2, :, :] * 0.587 + dark[2:3, :, :] * 0.114
#     light = img_nf
#     light = light[0:1, :, :] * 0.299 + light[1:2, :, :] * 0.587 + light[2:3, :, :] * 0.114  # 灰度化
#     noise = torch.abs(dark - light)
#     mask = torch.div(light, noise + 0.0001)  # torch.Size([4, 1, 240, 320])
#     # print('mask.shape', mask.shape)
#
#     batch_size = mask.shape[0]
#     height = mask.shape[2]
#     width = mask.shape[3]
#     mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
#     mask_max = mask_max.view(batch_size, 1, 1, 1)
#     mask_max = mask_max.repeat(1, 1, height, width)
#     mask = mask * 1.0 / (mask_max + 0.0001)
#
#     mask = torch.clamp(mask, min=0, max=1.0)
#     mask = mask.float()  # torch.Size([4, 1, 240, 320])
#
#     return mask
#
# img = Image.open(r'/mnt/disk2/data/stu010/lj/datasets/LISU/LISU_LLRGBD_real/val_rgb_l/411.png')
# img = img.resize((320, 240), Image.BILINEAR)
# img = np.array(img)
#
# img_dn = img
# img_dn = cv2.blur(img_dn, (5, 5))
# img_dn = torch.Tensor(img_dn).float().permute(2, 0, 1)
#
# img = Image.fromarray(img).convert('RGB')
# img = transforms.ToTensor(img)
#
# mask = gen_mask(img, img_dn)





