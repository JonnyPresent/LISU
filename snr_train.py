import argparse
import os
import sys
import time

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LLRGBD_real
from loss import ce_loss, pt_ssim_loss, pt_grad_loss, LossRetinex
from model.archs.low_light_transformer import low_light_transformer
from model.refinenetlw import rf_lw101, ResNetLW, Bottleneck
from models import LISU_DECOMP, LISU_JOINT
import numpy as np
from utils.optimisers import get_optimisers, get_lr_schedulers
from torch.nn import functional as F
from utils import utils
from PIL import Image
from torch.autograd import Variable
from model.fogpassfilter import FogPassFilter_conv1, FogPassFilter_res1
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import MeanReducer


def get_arguments(argv):

    parser = argparse.ArgumentParser(description="framework")

    parser.add_argument('--data_path', type=str, default=r'/mnt/disk2/data/stu010/lj/datasets/LISU/LISU_LLRGBD_real',
                        help='path to your dataset')
    parser.add_argument('--pretrained_model_path', type=str, default=r'ckpt/save/LISU_LLRGBD_real_best.pth.tar', help='')
    # parser.add_argument('--pretrained_resnet_path', type=str, default=r'ckpt/lisu-fifo/epoch350.pth', help='')

    parser.add_argument('--save_model_path', type=str, default=r'ckpt/snr-dec', help='path to save trained model')
    parser.add_argument('--visualization_path', type=str, default=r'cat_image/snr-dec')

    parser.add_argument('--num_epochs', type=int, default=400, help='Number of epochs to train for')
    parser.add_argument('--start_epoch', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of images in each batch')

    parser.add_argument('--num_classes', type=int, default=14, help='num of object classes (with void)')
    parser.add_argument("--lambda-fsm", type=float, default=0.0000001)
    parser.add_argument("--lambda-con", type=float, default=0.0001)

    return parser.parse_args(argv)


class SNRDec:
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.model_decomp = low_light_transformer()
        self.opt_decomp = torch.optim.Adam(self.model_decomp.parameters(), lr=0.001, betas=(0.95, 0.999))

        self.loss_decomp = LossRetinex()

        self.best_score = 0.3
        # self.load_model()

    def load_model(self):
        checkpoint_lisu = torch.load(self.args.pretrained_model_path)
        self.model_decomp.load_state_dict(checkpoint_lisu['state_dict_decomp'], strict=False)
        # self.model_enhance.load_state_dict(checkpoint_lisu['state_dict_enhance'], strict=False)
        # args.start_epoch = checkpoint['epoch']
        self.opt_decomp.load_state_dict(checkpoint_lisu['optimizer_decomp'])
        # self.opt_enhance.load_state_dict(checkpoint_lisu['optimizer_enhance'])
        # self.best_score = checkpoint_lisu['best_pred']
        print('加载lisu-decomp')


    def train(self, train_loader, eval_loader):
        max_score = self.best_score
        for epoch in range(self.args.start_epoch, self.args.num_epochs):
            loss = self.train_epoch(epoch, train_loader)
            print('epoch:{}, snr-loss:{}'.format(epoch, loss))
            self.save_model()

    def train_epoch(self, epoch, train_loader):
        self.model_decomp.train()
        loss = 0
        tbar = tqdm(train_loader)
        for i, (image, image2, image_dn, label, name) in enumerate(tbar):  # l lowlight highlight lable
            image = image.cuda()
            image2 = image2.cuda()
            # label = label.cuda()

            dark = image
            dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
            light = image_dn
            light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114  # 灰度化
            noise = torch.abs(dark - light)
            mask = torch.div(light, noise + 0.0001)

            batch_size = mask.shape[0]
            height = mask.shape[2]
            width = mask.shape[3]
            mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
            mask_max = mask_max.view(batch_size, 1, 1, 1)
            mask_max = mask_max.repeat(1, 1, height, width)
            mask = mask * 1.0 / (mask_max + 0.0001)

            mask = torch.clamp(mask, min=0, max=1.0)
            mask = mask.float()

            I, R = self.model_decomp(image, mask)
            I2, R2 = self.model_decomp(image2)

            I_3 = torch.cat((I, I, I), dim=1)
            I2_3 = torch.cat((I2, I2, I2), dim=1)
            recon_low = self.loss_decomp.recon(R, I_3, image)
            recon_high = self.loss_decomp.recon(R2, I2_3, image2)
            recon_low_2 = self.loss_decomp.recon(R2, I_3, image)
            recon_high_2 = self.loss_decomp.recon(R, I2_3, image2)
            smooth_i_low = self.loss_decomp.smooth(image, I)
            smooth_i_high = self.loss_decomp.smooth(image2, I2)
            max_i_low = self.loss_decomp.max_rgb_loss(image, I)
            max_i_high = self.loss_decomp.max_rgb_loss(image2, I2)

            loss = recon_low + recon_high + 0.01 * recon_low_2 + 0.01 * recon_high_2 \
                    + 0.5 * smooth_i_low + 0.5 * smooth_i_high + \
                    0.1 * max_i_low + 0.1 * max_i_high

            self.opt_decomp.zero_grad()
            loss.backward()
            self.opt_decomp.step()

            if epoch % 20 == 0 and i == 0 :
                self.visualization(epoch, image, I, R, image2, I2, R2)

        tbar.close()
        return loss.item()

    def evaluate(self, eval_loader, epoch):
        self.model_decomp.eval()
        lbls = []
        preds = []

        tbar = tqdm(eval_loader)
        for i, (image, image2, label, name) in enumerate(tbar):
            image = image.cuda()
            image2 = image2.cuda()

            I, R = self.model_decomp(image)
            I2, R2 = self.model_decomp(image2)


        return

    def save_model(self, epoch):
        if not os.path.isdir(self.args.save_model_path):
            os.makedirs(self.args.save_model_path)
        if epoch > 0 and epoch % 50 == 0:
            ckpt_path = os.path.join(self.args.save_model_path,
                                     'epoch{0}.pth'.format(epoch))
            save_state = {'model': self.model_enhance.state_dict(),
                          'optimizer_enhance': self.opt_enhance.state_dict(),
                          'epoch': epoch}
            torch.save(save_state, ckpt_path)
            print('保存断点')
        pass

    def visualization(self, epoch, image, I, R, image2, I2, R2):
        I_3 = torch.cat((I, I, I), dim=1)
        I2_3 = torch.cat((I2, I2, I2), dim=1)
        I_pow = torch.pow(I_3, 0.1)
        I_e = I_pow * R

        # 低光 R, I, image_h, R2, I2, 恢复低光
        cat_image = np.concatenate(
            [image[0, :, :, :].detach().cpu(), R[0, :, :, :].detach().cpu(), I_3[0, :, :, :].detach().cpu(),
             image2[0, :, :, :].detach().cpu(), R2[0, :, :, :].detach().cpu(), I2_3[0, :, :, :].detach().cpu(),
             I_e[0, :, :, :].detach().cpu()
             ], axis=2)
        cat_image = np.transpose(cat_image, (1, 2, 0))
        cat_image = np.clip(cat_image * 255.0, 0, 255.0).astype('uint8')

        im = Image.fromarray(cat_image)
        if not os.path.isdir(self.args.visualization_path):
            os.makedirs(self.args.visualization_path)
        filepath = os.path.join(self.args.visualization_path, f'val{epoch}.jpg')
        im.save(filepath)


if __name__ == '__main__':
    args = get_arguments(sys.argv[1:])
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    print('cuda_device', os.environ["CUDA_VISIBLE_DEVICES"])
    print('start_epoch', args.start_epoch)

    trainset = LLRGBD_real(args, mode='train')
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    valset = LLRGBD_real(args, mode='val')
    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

