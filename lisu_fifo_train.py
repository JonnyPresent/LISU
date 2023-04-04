import os
import time

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from loss import ce_loss, pt_ssim_loss, pt_grad_loss
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


def gram_matrix(tensor):
    d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


def make_list(x):
    """Returns the given input as a list."""
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]


def setup_optimisers_and_schedulers(args, model):
    optimisers = get_optimisers(
        model=model,
        enc_optim_type="sgd",
        enc_lr=6e-4,
        enc_weight_decay=1e-5,
        enc_momentum=0.9,
        dec_optim_type="sgd",
        dec_lr=6e-3,
        dec_weight_decay=1e-5,
        dec_momentum=0.9,
    )
    schedulers = get_lr_schedulers(
        enc_optim=optimisers[0],
        dec_optim=optimisers[1],
        enc_lr_gamma=0.5,
        dec_lr_gamma=0.5,
        enc_scheduler_type="multistep",
        dec_scheduler_type="multistep",
        epochs_per_stage=(100, 100, 100),
    )
    return optimisers, schedulers


class LisuFifo:
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.model_decomp = LISU_DECOMP().cuda()
        self.opt_decomp = torch.optim.Adam(self.model_decomp.parameters(), lr=0.001, betas=(0.95, 0.999))

        self.model_enhance = LISU_JOINT().cuda()
        self.opt_enhance = torch.optim.Adam(self.model_enhance.parameters(), lr=0.001, betas=(0.95, 0.999))
        self.log_m = torch.nn.LogSoftmax(dim=1)
        self.m = torch.nn.Softmax(dim=1)
        self.kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

        self.best_score = 0.4
        lr_fpf1 = 5e-4
        lr_fpf2 = 1e-3
        # if args.modeltrain == 'train':
        #     lr_fpf1 = 5e-4
        self.FogPassFilter1 = FogPassFilter_conv1(528).cuda()
        self.FogPassFilter1_optimizer = torch.optim.Adamax(
            [p for p in self.FogPassFilter1.parameters() if p.requires_grad == True],
            lr=lr_fpf1)
        self.FogPassFilter2 = FogPassFilter_res1(2080).cuda()
        self.FogPassFilter2_optimizer = torch.optim.Adamax(
            [p for p in self.FogPassFilter2.parameters() if p.requires_grad == True],
            lr=lr_fpf2)
        self.fogpassfilter_loss = losses.ContrastiveLoss(
            pos_margin=0.1,
            neg_margin=0.1,
            distance=CosineSimilarity(),
            reducer=MeanReducer()
        )
        # self.load_res_fifo()
        self.load_model()

    def load_model(self):
        checkpoint_lisu = torch.load(self.args.pretrained_model_path)
        self.model_decomp.load_state_dict(checkpoint_lisu['state_dict_decomp'], strict=False)
        self.model_enhance.load_state_dict(checkpoint_lisu['state_dict_enhance'], strict=False)
        # args.start_epoch = checkpoint['epoch']
        self.opt_decomp.load_state_dict(checkpoint_lisu['optimizer_decomp'])
        self.opt_enhance.load_state_dict(checkpoint_lisu['optimizer_enhance'])
        # self.best_score = checkpoint_lisu['best_pred']
        print('加载lisu')

        checkpoint = torch.load(self.args.pretrained_resnet_path)
        # self.model_enhance.load_state_dict(checkpoint['model'])
        self.FogPassFilter1.load_state_dict(checkpoint['fogpass1_state_dict'])
        self.FogPassFilter2.load_state_dict(checkpoint['fogpass2_state_dict'])
        print('加载fifo')

        # checkpoint = torch.load(self.args.pretrained_model_path)  # 只加载decomp
        # self.model_decomp.load_state_dict(checkpoint['state_dict_decomp'])
        # self.opt_decomp.load_state_dict(checkpoint['optimizer_decomp'])
        # print('加载decomp')

    def load_resnet(self):
        if self.args.pretrained_resnet_path is not None:
            if not os.path.isfile(self.args.pretrained_resnet_path):
                raise RuntimeError("=> no pretrained resnet model found at '{}'".format(self.args.pretrained_resnet_path))
            checkpoint = torch.load(self.args.pretrained_resnet_path)
            self.model_enhance.load_state_dict(checkpoint['model'])
            print('加载resnet,in_c=4')

        else:
            print('未加载resnet')

    def load_res_fifo(self):
        if self.args.pretrained_resnet_path is not None:
            if not os.path.isfile(self.args.pretrained_resnet_path):
                raise RuntimeError("=> no pretrained res_fifo model found at '{}'".format(self.args.pretrained_resnet_path))
            checkpoint = torch.load(self.args.pretrained_resnet_path)
            self.model_enhance.load_state_dict(checkpoint['model'])
            self.FogPassFilter1.load_state_dict(checkpoint['fogpass1_state_dict'])
            self.FogPassFilter2.load_state_dict(checkpoint['fogpass2_state_dict'])
            print('加载resnet_fifo,in_c=4')

        else:
            print('未加载resnet')

    def train(self, train_loader, eval_loader):
        # writer_name = '{}_{}'.format('LISU', str(time.strftime("%m-%d %H:%M:%S", time.localtime())))
        # writer = SummaryWriter(os.path.join('runs', writer_name))
        max_score = self.best_score
        for epoch in range(self.args.start_epoch, self.args.num_epochs):
            self.train_epoch(epoch, train_loader)
            acc, macc, miou = self.evaluate(eval_loader, epoch)
            print('epoch:{}, acc: {}, macc:{}, miou: {}, max_score: {}'.format(epoch, acc, macc, miou, max_score))

            # 保存模型
            if not os.path.isdir(self.args.save_model_path):
                os.makedirs(self.args.save_model_path)
            if miou > max_score:
                max_score = miou
                ckpt_path = os.path.join(self.args.save_model_path,
                                         'lisu-fifo_maxscore{1:.4f}_epoch{2}_{0}.pth'.format(epoch, max_score, self.args.num_epochs))
                save_state = {'model': self.model_enhance.state_dict(),
                              'optimizer_enhance': self.opt_enhance.state_dict(),
                              'fogpass1_state_dict': self.FogPassFilter1.state_dict(),
                              'fogpass1_opt_state_dict': self.FogPassFilter1_optimizer,
                              'fogpass2_state_dict': self.FogPassFilter2.state_dict(),
                              'fogpass2_opt_state_dict': self.FogPassFilter2_optimizer,
                              'epoch': epoch}
                torch.save(save_state, ckpt_path)
                print('保存best:')
            elif epoch > 0 and epoch % 50 == 0:
                ckpt_path = os.path.join(self.args.save_model_path,
                                         'epoch{0}.pth'.format(epoch, max_score))
                save_state = {'model': self.model_enhance.state_dict(),
                              'optimizer_enhance': self.opt_enhance.state_dict(),
                              'fogpass1_state_dict': self.FogPassFilter1.state_dict(),
                              'fogpass1_opt_state_dict': self.FogPassFilter1_optimizer,
                              'fogpass2_state_dict': self.FogPassFilter2.state_dict(),
                              'fogpass2_opt_state_dict': self.FogPassFilter2_optimizer,
                              'epoch': epoch}
                torch.save(save_state, ckpt_path)
                print('保存断点')

    def train_epoch(self, epoch, train_loader):
        self.model_decomp.eval()

        tbar = tqdm(train_loader)
        loss_list = []
        fpf_loss, loss, loss_r, loss_s = 0, 0, 0, 0
        for i, (image, image2, label, name) in enumerate(tbar):  # l lowlight highlight lable
            image = image.cuda()
            image2 = image2.cuda()
            label = label.cuda()
            # image_size = np.array(image.shape)

            I, R = self.model_decomp(image)
            I2, R2 = self.model_decomp(image2)

            input_i_r = torch.cat((I.detach(), R.detach()), dim=1)
            input_i2_r2 = torch.cat((I2.detach(), R2.detach()), dim=1)
            # Variable(torch.cat((I.detach(), R.detach()), dim=1)).cuda()  # 只训练resnet
            # _, _, _, _, _, feature5 = self.model_rf(input_i_r)
            # interp = torch.nn.Upsample(size=(image_size[2], image_size[3]), mode='bilinear', align_corners=True)
            # pred = interp(feature5)
            #
            # loss = ce_loss(pred, label)
            # for opt in self.opts_rf:
            #     opt.zero_grad()
            # loss.backward()
            # for opt in self.opts_rf:
            #     opt.step()
            #
            # with torch.no_grad():
            #     loss_list.append(loss.item())
            #
            # tbar.set_description('seg_loss_mean: {0:.3}, seg_loss: {1:.3}'
            #                      .format(np.mean(loss_list), loss))
            fpf_loss = self.train_fifo_freeze_seg(input_i_r, input_i2_r2)
            if epoch > 100:
                loss, loss_r, loss_s = self.train_seg_freeze_fifo(input_i_r, input_i2_r2, label, R2)
            self.FogPassFilter1_optimizer.step()
            self.FogPassFilter2_optimizer.step()
        print('fpf_loss:', fpf_loss, 'loss:', loss, 'loss_r:', loss_r, 'loss_s:', loss_s)
        tbar.close()
        return

    def train_fifo_freeze_seg(self, input_i_r, input_i2_r2):
        self.model_enhance.eval()
        for param in self.model_enhance.parameters():
            param.requires_grad = False
        for param in self.FogPassFilter1.parameters():
            param.requires_grad = True
        for param in self.FogPassFilter2.parameters():
            param.requires_grad = True

        # input_i_r = Variable(torch.cat((I.detach(), R.detach()), dim=1)).cuda()
        # input_i2_r2 = Variable(torch.cat((I2.detach(), R2.detach()), dim=1)).cuda()
        # feature_sf0, feature_sf1, feature_sf2, feature_sf3, feature_sf4, feature_sf5 = self.model_enhance(input_i_r)
        # feature_cw0, feature_cw1, feature_cw2, feature_cw3, feature_cw4, feature_cw5 = self.model_enhance(input_i2_r2)
        _, _, feature_sf0, feature_sf1 = self.model_enhance(input_i_r)
        _, _, feature_cw0, feature_cw1 = self.model_enhance(input_i2_r2)
        fsm_weights = {'layer0': 0.5, 'layer1': 0.5}
        low_features = {'layer0': feature_sf0, 'layer1': feature_sf1}
        high_features = {'layer0': feature_cw0, 'layer1': feature_cw1}

        total_fpf_loss = 0

        for idx, layer in enumerate(fsm_weights):
            low_feature = low_features[layer]
            high_feature = high_features[layer]
            if idx == 0:
                fogpassfilter = self.FogPassFilter1
                fogpassfilter_optimizer = self.FogPassFilter1_optimizer
            elif idx == 1:
                fogpassfilter = self.FogPassFilter2
                fogpassfilter_optimizer = self.FogPassFilter2_optimizer
            else:
                raise KeyError('no such layer id')

            fogpassfilter.train()
            fogpassfilter_optimizer.zero_grad()

            low_gram = [0] * self.args.batch_size
            high_gram = [0] * self.args.batch_size
            vector_low_gram = [0] * self.args.batch_size
            vector_high_gram = [0] * self.args.batch_size
            fog_factor_low = [0] * self.args.batch_size
            fog_factor_high = [0] * self.args.batch_size

            for batch_idx in range(self.args.batch_size):
                low_gram[batch_idx] = gram_matrix(low_feature[batch_idx])
                high_gram[batch_idx] = gram_matrix(high_feature[batch_idx])
                # print(low_gram[batch_idx].shape)

                vector_low_gram[batch_idx] = Variable(low_gram[batch_idx][torch.triu(
                    torch.ones(low_gram[batch_idx].size()[0], low_gram[batch_idx].size()[1])) == 1],
                                                      requires_grad=True)
                vector_high_gram[batch_idx] = Variable(high_gram[batch_idx][torch.triu(
                    torch.ones(high_gram[batch_idx].size()[0], high_gram[batch_idx].size()[1])) == 1],
                                                       requires_grad=True)
                # print(vector_low_gram[batch_idx].shape)

                fog_factor_low[batch_idx] = fogpassfilter(vector_low_gram[batch_idx])
                fog_factor_high[batch_idx] = fogpassfilter(vector_high_gram[batch_idx])

            fog_factor_embeddings = torch.cat((torch.unsqueeze(fog_factor_low[0], 0), torch.unsqueeze(fog_factor_high[0], 0),
                                               torch.unsqueeze(fog_factor_low[1], 0),torch.unsqueeze(fog_factor_high[1], 0),
                                               torch.unsqueeze(fog_factor_low[2], 0),torch.unsqueeze(fog_factor_high[2], 0),
                                               torch.unsqueeze(fog_factor_low[3], 0),torch.unsqueeze(fog_factor_high[3], 0),
                                               ), 0)
            # print(fog_factor_embeddings.shape)

            fog_factor_embeddings_norm = torch.norm(fog_factor_embeddings, p=2, dim=1).detach()
            size_fog_factor = fog_factor_embeddings.size()
            fog_factor_embeddings = fog_factor_embeddings.div(
                fog_factor_embeddings_norm.expand(size_fog_factor[1], 8).t())
            fog_factor_labels = torch.LongTensor([0, 1, 0, 1, 0, 1, 0, 1])
            fog_pass_filter_loss = self.fogpassfilter_loss(fog_factor_embeddings, fog_factor_labels)

            total_fpf_loss += fog_pass_filter_loss
        total_fpf_loss.backward(retain_graph=False)
        return total_fpf_loss.item()

    def train_seg_freeze_fifo(self, input_i_r, input_i2_r2, label, R2):
        self.model_enhance.train()
        # interp = torch.nn.Upsample(size=(image_size[2], image_size[3]), mode='bilinear', align_corners=True)

        for param in self.model_enhance.parameters():
            param.requires_grad = True
        for param in self.FogPassFilter1.parameters():
            param.requires_grad = False
        for param in self.FogPassFilter2.parameters():
            param.requires_grad = False

        # input_i_r = Variable(torch.cat((I.detach(), R.detach()), dim=1)).cuda()
        # input_i2_r2 = Variable(torch.cat((I2.detach(), R2.detach()), dim=1)).cuda()
        R_hat, smap_l, feature_sf0, feature_sf1 = self.model_enhance(input_i_r)
        _, smap_h, feature_cw0, feature_cw1 = self.model_enhance(input_i2_r2)
        fsm_weights = {'layer0': 0.5, 'layer1': 0.5}
        low_features = {'layer0': feature_sf0, 'layer1': feature_sf1}
        high_features = {'layer0': feature_cw0, 'layer1': feature_cw1}

        # pred_sf5 = utils.reverse_one_hot(F.softmax(smap_l, 1))
        # pred_cw5 = utils.reverse_one_hot(F.softmax(smap_h, 1))
        feature_sf5_logsoftmax = self.log_m(smap_l)
        feature_cw5_softmax = self.m(smap_h)

        loss_seg_sf = ce_loss(smap_l, label)
        loss_seg_cw = ce_loss(smap_h, label)
        loss_con = self.kl_loss(feature_sf5_logsoftmax, feature_cw5_softmax)
        loss_fsm = 0

        for idx, layer in enumerate(fsm_weights):
            low_feature = low_features[layer]
            high_feature = high_features[layer]
            layer_fsm_loss = 0

            na, da, ha, wa = low_feature.size()
            nb, db, hb, wb = high_feature.size()

            if idx == 0:
                fogpassfilter = self.FogPassFilter1
                fogpassfilter_optimizer = self.FogPassFilter1_optimizer
            elif idx == 1:
                fogpassfilter = self.FogPassFilter2
                fogpassfilter_optimizer = self.FogPassFilter2_optimizer
            fogpassfilter.eval()

            for batch_idx in range(self.args.batch_size):
                low_gram = gram_matrix(low_feature[batch_idx])
                high_gram = gram_matrix(high_feature[batch_idx])

                # low_gram = low_gram * (hb * wb) / (ha * wa)  #

                vector_low_gram = low_gram[torch.triu(
                    torch.ones(low_gram.size()[0], low_gram.size()[1])).requires_grad_() == 1].requires_grad_()
                vector_high_gram = high_gram[torch.triu(
                    torch.ones(high_gram.size()[0], high_gram.size()[1])).requires_grad_() == 1].requires_grad_()
                fog_factor_a = fogpassfilter(vector_low_gram)
                fog_factor_b = fogpassfilter(vector_high_gram)
                half = int(fog_factor_b.shape[0] / 2)

                layer_fsm_loss += fsm_weights[layer] * torch.mean(
                    (fog_factor_b / (hb * wb) - fog_factor_a / (ha * wa)) ** 2) / half / high_gram.size(0)

            loss_fsm += layer_fsm_loss / 4.

        recon_r = torch.mean(torch.pow((R_hat - R2.detach()), 2))
        ssim_r = pt_ssim_loss(R_hat, R2.detach())
        grad_r = pt_grad_loss(R_hat, R2.detach())

        loss_r = recon_r + ssim_r + grad_r
        loss_s = loss_seg_sf + loss_seg_cw + self.args.lambda_fsm * loss_fsm + self.args.lambda_con * loss_con
        loss = 0.2*loss_r + 0.8*loss_s

        # for opt in self.opt_enhance:
        #     opt.zero_grad()
        # loss.backward()
        # for opt in self.opt_enhance:
        #     opt.step()
        self.opt_enhance.zero_grad()
        loss.backward()
        self.opt_enhance.step()
        return loss.item(), loss_r.item(), loss_s.item()

    def evaluate(self, eval_loader, epoch):
        self.model_decomp.eval()
        self.model_enhance.eval()
        lbls = []
        preds = []

        tbar = tqdm(eval_loader)
        for i, (image, image2, label, name) in enumerate(tbar):
            image = image.cuda()
            image_size = np.array(image.shape)

            I, R = self.model_decomp(image)
            input_i_r = torch.cat((I.detach(), R.detach()), dim=1)
            R_hat, smap_l, feature_sf0, feature_sf1 = self.model_enhance(input_i_r)
            # interp = torch.nn.Upsample(size=(image_size[2], image_size[3]), mode='bilinear', align_corners=True)
            # smap = interp(feature5)
            smap = F.softmax(smap_l, 1)
            smap_oh = utils.reverse_one_hot(smap)
            for l, p in zip(label.data.cpu().numpy(), smap_oh.data.cpu().numpy()):
                lbls.append(l)
                preds.append(p)

            if epoch % 50 == 0 and i % 1000 == 0:
                self.visualization(epoch, image, I, R, smap_oh, label)

        acc, acc_cls, _, _, iou = utils.label_accuracy_score(lbls, preds, self.args.num_classes)
        miou = np.sum(np.nanmean(iou[1:]))
        macc = np.sum(np.nanmean(acc_cls[1:]))
        return acc, macc, miou

    def visualization(self, epoch, image, I, R, sout, label):
        sout = sout[0, :, :]
        sout = utils.colorize(sout).numpy()
        sout = np.transpose(sout, (2, 0, 1))

        lbl = label[0, :, :]
        lbl = utils.colorize(lbl).numpy()
        lbl = np.transpose(lbl, (2, 0, 1))


        I_3 = torch.cat((I, I, I), dim=1)
        # 低光 R, I, R2, I2, 恢复R, 恢复低光，label, 分割结果
        # 低光 R, I, label, 分割结果
        cat_image = np.concatenate(
            [image[0, :, :, :].detach().cpu(), R[0, :, :, :].detach().cpu(), I_3[0, :, :, :].detach().cpu(),
             lbl, sout], axis=2)
        cat_image = np.transpose(cat_image, (1, 2, 0))
        cat_image = np.clip(cat_image * 255.0, 0, 255.0).astype('uint8')

        im = Image.fromarray(cat_image)
        if not os.path.isdir(self.args.visualization_path):
            os.makedirs(self.args.visualization_path)
        filepath = os.path.join(self.args.visualization_path, f'val{epoch}.jpg')
        im.save(filepath)
