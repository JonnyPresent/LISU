"""
@FileName: main.py
@Time    : 7/16/2020
@Author  : Ning Zhang
@GitHub: https://github.com/noahzn
"""

import os
from tqdm import tqdm
import csv
import matplotlib
from tensorboardX import SummaryWriter

matplotlib.use('Agg')
from loss import *
from dataset import LLRGBD_real, LLRGBD_synthetic
from PIL import Image
from utils import utils

import models
import argparse
import sys
import time

from model.fogpassfilter import FogPassFilter_conv1, FogPassFilter_res1
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import MeanReducer

from torch.autograd import Variable
from torch.nn import functional as F

from torchvision import utils as vutils

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(profile='full')


# torch.backends.cudnn.benchmark = True


def gram_matrix(tensor):
    d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


def train_and_val(args, Decomp, opt_Decomp, Enhance, opt_Enhance, trainloader, valloader, best_pred=0.0):
    writer_name = '{}_{}'.format('LISU', str(time.strftime("%m-%d %H:%M:%S", time.localtime())))
    writer = SummaryWriter(os.path.join('runs', writer_name))
    step = 0

    opt_Enhance.zero_grad()
    opt_Decomp.zero_grad()

    loss_decomp = LossRetinex()
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
    log_m = torch.nn.LogSoftmax(dim=1)
    m = torch.nn.Softmax(dim=1)

    lr_fpf1 = 1e-3
    lr_fpf2 = 1e-3
    FogPassFilter1 = FogPassFilter_conv1(528)
    FogPassFilter1_optimizer = torch.optim.Adamax([p for p in FogPassFilter1.parameters() if p.requires_grad == True],
                                                  lr=lr_fpf1)
    FogPassFilter1.cuda()
    FogPassFilter2 = FogPassFilter_res1(2080)
    FogPassFilter2_optimizer = torch.optim.Adamax([p for p in FogPassFilter2.parameters() if p.requires_grad == True],
                                                  lr=lr_fpf2)
    FogPassFilter2.cuda()

    fogpassfilter_loss = losses.ContrastiveLoss(
        pos_margin=0.1,
        neg_margin=0.1,
        distance=CosineSimilarity(),
        reducer=MeanReducer()
    )

    for epoch in range(args.start_epoch, args.num_epochs):
        lr = utils.poly_lr_scheduler(opt_Enhance, 0.001, epoch, max_iter=args.num_epochs)
        print('epoch:', epoch, 'best:', best_pred, ' lr_enhance:', lr, 'lr_fpf1:', lr_fpf1, 'lr_fpf2:', lr_fpf2)
        loss_record = []
        miou_record = []

        # if epoch > 150:
        #     for p in Decomp.parameters():
        #         p.requires_grad = False
        # else:
        #     Decomp.train()

        Decomp.eval()
        tbar = tqdm(trainloader)

        for i, (image, image2, label, name) in enumerate(tbar):  # l lowlight highlight lable
            train_loss = 0.0
            image = image.cuda()
            label = label.cuda()
            image2 = image2.cuda()

            I, R = Decomp(image)
            I2, R2 = Decomp(image2)

            # I_3 = torch.cat((I, I, I), dim=1)
            # I2_3 = torch.cat((I2, I2, I2), dim=1)
            # recon_low = loss_decomp.recon(R, I_3, image)
            # recon_high = loss_decomp.recon(R2, I2_3, image2)
            # recon_low_2 = loss_decomp.recon(R2, I_3, image)
            # recon_high_2 = loss_decomp.recon(R, I2_3, image2)
            # smooth_i_low = loss_decomp.smooth(image, I)
            # smooth_i_high = loss_decomp.smooth(image2, I2)
            # max_i_low = loss_decomp.max_rgb_loss(image, I)
            # max_i_high = loss_decomp.max_rgb_loss(image2, I2)
            #
            # loss1 = recon_low + recon_high + 0.01 * recon_low_2 + 0.01 * recon_high_2 \
            #         + 0.5 * smooth_i_low + 0.5 * smooth_i_high + \
            #         0.1 * max_i_low + 0.1 * max_i_high
            #
            # # l# 后面epoch 只训练joint net
            # if epoch > -1:
            #     opt_Decomp.zero_grad()
            # else:
            #     opt_Decomp.zero_grad()
            #     loss1.backward()
            #     opt_Decomp.step()
            # writer.add_scalar("decom_loss", loss1, step)

            # train fog-pass filtering module
            # freeze the parameters of segmentation network
            # -----------------------------------------------------------
            Enhance.eval()
            for param in Enhance.parameters():
                param.requires_grad = False
            for param in FogPassFilter1.parameters():
                param.requires_grad = True
            for param in FogPassFilter2.parameters():
                param.requires_grad = True

            input_i_r = Variable(torch.cat((I.detach(), R.detach()), dim=1)).cuda()
            input_i2_r2 = Variable(torch.cat((I2.detach(), R2.detach()), dim=1)).cuda()
            smap, feature_low1, feature_low2 = Enhance(input_i_r)
            smap_n, feature_high1, feature_high2 = Enhance(input_i2_r2)
            fsm_weights = {'layer0': 0.5, 'layer1': 0.5}
            low_features = {'layer0': feature_low1, 'layer1': feature_low2}
            high_features = {'layer0': feature_high1, 'layer1': feature_high2}
            total_fpf_loss = 0

            for idx, layer in enumerate(fsm_weights):
                low_feature = low_features[layer]
                high_feature = high_features[layer]
                if idx == 0:
                    fogpassfilter = FogPassFilter1
                    fogpassfilter_optimizer = FogPassFilter1_optimizer
                elif idx == 1:
                    fogpassfilter = FogPassFilter2
                    fogpassfilter_optimizer = FogPassFilter2_optimizer

                fogpassfilter.train()
                fogpassfilter_optimizer.zero_grad()

                low_gram = [0] * args.batch_size
                high_gram = [0] * args.batch_size
                vector_low_gram = [0] * args.batch_size
                vector_high_gram = [0] * args.batch_size
                fog_factor_low = [0] * args.batch_size
                fog_factor_high = [0] * args.batch_size

                for batch_idx in range(args.batch_size):
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

                fog_factor_embeddings = torch.cat((torch.unsqueeze(fog_factor_low[0], 0),
                                                   torch.unsqueeze(fog_factor_high[0], 0),
                                                   torch.unsqueeze(fog_factor_low[1], 0),
                                                   torch.unsqueeze(fog_factor_high[1], 0)), 0)
                # print(fog_factor_embeddings.shape)

                fog_factor_embeddings_norm = torch.norm(fog_factor_embeddings, p=2, dim=1).detach()
                size_fog_factor = fog_factor_embeddings.size()
                fog_factor_embeddings = fog_factor_embeddings.div(
                    fog_factor_embeddings_norm.expand(size_fog_factor[1], 4).t())
                fog_factor_labels = torch.LongTensor([0, 1, 0, 1])
                fog_pass_filter_loss = fogpassfilter_loss(fog_factor_embeddings, fog_factor_labels)

                total_fpf_loss += fog_pass_filter_loss
            total_fpf_loss.backward(retain_graph=False)
            FogPassFilter1_optimizer.step()
            FogPassFilter2_optimizer.step()
            # ----------------------------------------------------------

            # train segmentation network
            # freeze the parameters of fog pass filtering modules
            # l#I R拼接 作为输入
            Enhance.train()
            loss_fsm = 0
            outfeature_l_logsoftmax = log_m(smap)
            outfeature_h_softmax = m(smap_n)
            loss_con = kl_loss(outfeature_l_logsoftmax, outfeature_h_softmax)

            for param in Enhance.parameters():
                param.requires_grad = True
            for param in FogPassFilter1.parameters():
                param.requires_grad = False
            for param in FogPassFilter2.parameters():
                param.requires_grad = False

            for idx, layer in enumerate(fsm_weights):
                low_feature = low_features[layer]
                high_feature = high_features[layer]
                layer_fsm_loss = 0

                na, da, ha, wa = low_feature.size()
                nb, db, hb, wb = high_feature.size()

                if idx == 0:
                    fogpassfilter = FogPassFilter1
                    fogpassfilter_optimizer = FogPassFilter1_optimizer
                elif idx == 1:
                    fogpassfilter = FogPassFilter2
                    fogpassfilter_optimizer = FogPassFilter2_optimizer
                fogpassfilter.eval()

                for batch_idx in range(args.batch_size):
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

            # -----------------------------------------------
            loss_s = ce_loss(smap_n, label) + ce_loss(smap,
                                                      label) + args.lambda_fsm * loss_fsm + args.lambda_con * loss_con
            loss = loss_s

            opt_Enhance.zero_grad()
            loss.backward()
            opt_Enhance.step()
            step += 1

            # train_loss += loss1.item() + loss.item()
            train_loss += loss.item()

            writer.add_scalar('seg_loss', loss, step)
            writer.add_scalar('train_loss', train_loss, step)
            loss_record.append(train_loss)

            # tbar.set_description('TrainLoss: {0:.3} mIoU: {1:.3} S: {2:.3}'.format(

            # tbar.set_description('TrainLoss: {0:.3}, seg_loss: {1:.3}, decom_loss:{2:.3},'
            #                      .format(np.mean(loss_record), loss_s, loss1))
            tbar.set_description('TrainLoss: {0:.3}, seg_loss: {1:.3}'
                                 .format(np.mean(loss_record), loss_s))

            # if i % 600 == 0:
            #     I_pow = torch.pow(I_3, 0.1)
            #     # I_e = I_pow * R_hat
            #     I_e = I_pow
            #     sout = utils.reverse_one_hot(smap)
            #     sout = sout[0, :, :]
            #     sout = utils.colorize(sout).numpy()
            #     sout = np.transpose(sout, (2, 0, 1))
            #
            #     lbl = label[0, :, :]
            #     lbl = utils.colorize(lbl).numpy()
            #     lbl = np.transpose(lbl, (2, 0, 1))
            #     cat_image = np.concatenate(
            #         [image[0, :, :, :].detach().cpu(), R[0, :, :, :].detach().cpu(), I_3[0, :, :, :].detach().cpu(),
            #          R2[0, :, :, :].detach().cpu(), I2_3[0, :, :, :].detach().cpu(), R_hat[0, :, :, :].detach().cpu(),
            #          I_e[0, :, :, :].detach().cpu(), lbl, sout], axis=2)
            #     cat_image = np.transpose(cat_image, (1, 2, 0))
            #     cat_image = np.clip(cat_image * 255.0, 0, 255.0).astype('uint8')
            #
            #     im = Image.fromarray(cat_image)
            #     # filepath = os.path.join('/path/to/folder', 'train_%d_%d.png' %(epoch, i))
            #     filepath = os.path.join('cat_image/', 'train_%d_%d.png' % (epoch, i))
            #
            #     im.save(filepath[:-4] + '.jpg')

        tbar.close()
        # loss_train_mean = np.mean(loss_record)
        # iou_train = np.nanmean(miou_record)

        # writer.add_scalar('iou_train', iou_train, epoch)
        # writer.add_scalar('loss_epoch_train', float(loss_train_mean), epoch)

        if epoch % args.validation_step == 0:
            acc, macc, miou = val(args, Decomp, opt_Decomp, Enhance, opt_Enhance, trainloader, valloader, epoch)
            writer.add_scalar('acc_val', acc, epoch)
            writer.add_scalar('miou_val', miou, epoch)

            if not os.path.isdir(args.save_model_path):
                os.makedirs(args.save_model_path)

            # print('iou: {}, acc: {}, best: {}, new: {}'.format(iou, acc, best_pred, (iou + acc) / 2))
            print(' acc: {}, macc:{}, miou: {}, best: {}'.format(acc, macc, miou, best_pred))

            if miou > best_pred:  # miou + acc
                best_pred = miou
                # 保存模型
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict_decomp': Decomp.state_dict(),
                    'optimizer_decomp': opt_Decomp.state_dict(),
                    'state_dict_enhance': Enhance.state_dict(),
                    'optimizer_enhance': opt_Enhance.state_dict(),
                    'best_pred': miou,
                }, is_best=True, filename='epoch{}best{}.pth'.format(epoch, best_pred))
            elif epoch > 0 and epoch % 50 == 0:  # 每20轮保存一次
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict_decomp': Decomp.state_dict(),
                    'optimizer_decomp': opt_Decomp.state_dict(),
                    'state_dict_enhance': Enhance.state_dict(),
                    'optimizer_enhance': opt_Enhance.state_dict(),
                    'best_pred': miou,
                }, is_best=False, filename='epoch{0}.pth'.format(epoch))


def val(args, Decomp, opt_Decomp, Enhance, opt_Enhance, trainloader, valloader, epoch=0):
    print('start val!')

    with torch.no_grad():
        Decomp.eval()
        Enhance.eval()
        loss_decomp = LossRetinex()
        lbls = []
        preds = []
        loss_record = []
        iou_record = []

        tbar = tqdm(valloader)
        for i, (image, image2, label, name) in enumerate(tbar):
            image = image.cuda()
            image2 = image2.cuda()
            label = label.cuda()

            # label_sd1 = label_sd1.cuda()
            # label_sd2 = label_sd2.cuda()
            # label_sd3 = label_sd3.cuda()

            I, R = Decomp(image)
            I2, R2 = Decomp(image2)
            I_3 = torch.cat((I, I, I), dim=1)
            I2_3 = torch.cat((I2, I2, I2), dim=1)

            smap, _, _ = Enhance(torch.cat((I.detach(), R.detach()), dim=1))

            # recon1 = loss_decomp.recon(R, I, image)
            # smooth_i = illumination_smooth_loss(R, I)
            # max_i = loss_decomp.max_rgb_loss(image, I)
            # loss1 = recon1 + 0.05*smooth_i + 0.02*max_i
            # loss_s = ce_loss(smap, label)
            #
            # loss = loss1 + loss_s
            # loss_record.append(loss.item())
            smap = F.softmax(smap, 1)
            smap_oh = utils.reverse_one_hot(smap)

            for l, p in zip(label.data.cpu().numpy(), smap_oh.data.cpu().numpy()):
                lbls.append(l)
                preds.append(p)

            if i % 100 == 0 and epoch % 50 == 0:
                I_pow = torch.pow(I_3, 0.1)
                # I_e = I_pow * R_hat
                I_e = I_pow
                sout = utils.reverse_one_hot(smap)
                sout = sout[0, :, :]
                sout = utils.colorize(sout).numpy()

                sout = np.transpose(sout, (2, 0, 1))

                lbl = label[0, :, :]
                lbl = utils.colorize(lbl).numpy()
                lbl = np.transpose(lbl, (2, 0, 1))
                # 低光 R, I, R2, I2, -, I_e, label, 分割结果
                cat_image = np.concatenate(
                    [image[0, :, :, :].detach().cpu(), R[0, :, :, :].detach().cpu(), I_3[0, :, :, :].detach().cpu(),
                     image2[0, :, :, :].detach().cpu(), R2[0, :, :, :].detach().cpu(), I2_3[0, :, :, :].detach().cpu(),
                     lbl, sout], axis=2)
                cat_image = np.transpose(cat_image, (1, 2, 0))
                cat_image = np.clip(cat_image * 255.0, 0, 255.0).astype('uint8')

                im = Image.fromarray(cat_image)  # 低光，I,R,I_3,R2,I2_3 R_hat,I_e I*R,lbl, sout
                # 低光 R, I, R2, I2, 恢复R, 恢复低光，label, 分割结果
                # filepath = os.path.join('/path/to/save/folder', 'val_%d.png' % i)
                filepath = os.path.join('cat_image/lisu-fifo-4', f'val{epoch}_{i}.jpg')
                im.save(filepath)

        acc, acc_cls, _, _, iou = utils.label_accuracy_score(lbls, preds, args.num_classes)
        miou = np.sum(np.nanmean(iou[1:]))
        macc = np.sum(np.nanmean(acc_cls[1:]))
    return acc, macc, miou


def output(args, Net, Net2, valloader):
    print('start output!')

    with torch.no_grad():
        Net.eval()
        Net2.eval()

        tbar = tqdm(valloader)
        for i, (image, image2, label, name) in enumerate(tbar):
            image = image.cuda()
            label = label.cuda()
            image2 = image2.cuda()

            if args.multiple_GPUs:
                output = output[0]

            else:
                I, R = Net(image)
                I2, R2 = Net(image2)
                R_hat, smap = Net2(torch.cat((I.detach(), R.detach()), dim=1))

                sout = utils.reverse_one_hot(smap)
                sout = sout[0, :, :]
                sout = utils.colorize(sout).numpy()
                sout = np.transpose(sout, (2, 0, 1))

                lbl = label[0, :, :]
                lbl = utils.colorize(lbl).numpy()
                lbl = np.transpose(lbl, (2, 0, 1))

                I_3 = torch.cat((I, I, I), dim=1)
                I2_3 = torch.cat((I2, I2, I2), dim=1)
                I_pow = torch.pow(I_3, 0.3)
                I_e = I_pow * R_hat

                output_list = [I_e, I_3, I2_3, R, R_hat, sout, lbl, R2]
                output_list_name = ['enhanced', 'i', 'i2', 'r', 'r_hat', 'seg', 'lbl', 'r2']

                for x in range(len(output_list)):
                    if x == 5:
                        cat_image = np.concatenate([sout], axis=2)
                    elif x == 6:
                        cat_image = np.concatenate([lbl], axis=2)
                    else:
                        cat_image = np.concatenate([output_list[x][0, :, :, :].detach().cpu()], axis=2)
                    filepath = os.path.join('/path/to/output_lisujoint_t', '%s_%s.png' % (name[0], output_list_name[x]))

                    cat_image = np.transpose(cat_image, (1, 2, 0))
                    cat_image = np.clip(cat_image * 255.0, 0, 255.0).astype('uint8')

                    im = Image.fromarray(cat_image)
                    im.save(filepath[:-4] + '.png')


def evaluation(args, Net, Net2, valloader):
    print('start evaluation!')

    folder_test = r'E:\10-code\dataset\lisu\LISU_test_lj'

    with torch.no_grad():
        Net.eval()
        Net2.eval()
        lbls = []
        preds = []

        tbar = tqdm(valloader)
        for i, (image, image2, label, name) in enumerate(tbar):
            image = image.cuda()

            if args.multiple_GPUs:
                output = output[0]

            else:
                I, R = Net(image)  # [1, 1, 240, 320]  [1, 3, 240, 320]
                R_hat, smap, _, _ = Net2(torch.cat((I.detach(), R.detach()), dim=1))  # smap [1, 14, 240, 320] 14-class

                # vutils.save_image(I, 'ilu.png')
                # vutils.save_image(R, 'ref.png')
                # print(f'i shape {I.shape} ')
                # print(f'r shape {R.shape} ')
                # print(f'smap shape {smap.shape} ')

                smap_oh = utils.reverse_one_hot(smap)  # [1, 240, 320]
                # print(f'smap_oh shape {smap_oh.shape} ')

                # if i == 0:
                #     vutils.save_image(smap_oh[0], 'smap_oh.png')  # gray picture

                for l, p in zip(label.data.cpu().numpy(), smap_oh.data.cpu().numpy()):
                    lbls.append(l)
                    preds.append(p)

        # acc, acc_cls, iou, _, iu = utils.label_accuracy_score(lbls, preds, args.num_classes)

        acc, acc_cls, _, _, iou = utils.label_accuracy_score(lbls, preds, args.num_classes)  # ll
        miou = np.sum(np.nanmean(iou[1:]))  # l mIOU  0: background; 去除background
        macc = np.sum(np.nanmean(acc_cls[1:]))  # l mAcc

        print('[lisu evaluate] oa:{0}, mAcc:{1}, mIou:{2}'.format(acc, macc, miou))  # acc: oa;
        print('acc_cls:{0}\niou:{1}'.format(acc_cls.tolist(), iou.tolist()))  # ll
        #
        # print(iu)
        # print(np.sum(np.nanmean(iu[1:])))  # l mIOU  0: background
        # print(acc)  # l oa
        # print(acc_cls)
        # print(np.sum(np.nanmean(acc_cls[1:])))  # l mAcc


def main(argv):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to train for')
    parser.add_argument('--checkpoint_step', type=int, default=5, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
    parser.add_argument('--crop_height', type=int, default=576, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=768, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of images in each batch')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--print_freq', type=int, default=600, help='print freq')
    parser.add_argument('--num_classes', type=int, default=14, help='num of object classes (with void)')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    # parser.add_argument('--pretrained_model_path', type=str, default='/path/to/LISU_LLRGBD_real_best.pth.tar', help='saved model')
    # parser.add_argument('--save_model_path', type=str, default='./checkpoints', help='path to save trained model')

    #  ll
    parser.add_argument('--data_path', type=str, default=r'/mnt/disk2/data/stu010/lj/datasets/LISU/LISU_LLRGBD_real',
                        help='path to your dataset')
    parser.add_argument('--pretrained_model_path', type=str, default=r'ckpt/save/LISU_LLRGBD_real_best.pth.tar', help='')
    # parser.add_argument('--pretrained_model_path', type=str, default=None, help='saved model')
    parser.add_argument('--mode', type=str, default='train_and_val', help='train_and_val|output|evaluation')
    parser.add_argument('--cuda', type=str, default='1', help='GPU id used for training')
    parser.add_argument('--save_model_path', type=str, default='ckpt', help='path to save trained model')
    parser.add_argument('--num_workers', type=int, default=2, help='num of workers')
    # parser.add_argument('--multiple-GPUs', default=True, help='val使用')
    parser.add_argument('--multiple-GPUs', default=False, help='')
    parser.add_argument('--start_epoch', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--lambda_fsm', type=int, default=0.00001, help='')
    parser.add_argument('--lambda_con', type=int, default=0.0001, help='')

    args = parser.parse_args(argv)

    # gpu id
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    print(f"CUDA_VISIBLE_DEVICES:{args.cuda}")

    import torch
    from torch.utils.data import DataLoader

    # random seed
    np.random.seed(args.seed)  # cpu vars
    torch.manual_seed(args.seed)  # cpu  vars
    torch.cuda.manual_seed_all(args.seed)

    Net = models.LISU_DECOMP().cuda()
    Net2 = models.LISU_JOINT().cuda()

    # optimizer
    opt_Net = torch.optim.Adam(Net.parameters(), lr=0.001, betas=(0.95, 0.999))
    opt_Net2 = torch.optim.Adam(Net2.parameters(), lr=0.001, betas=(0.95, 0.999))

    if args.pretrained_model_path is not None:
        if not os.path.isfile(args.pretrained_model_path):
            raise RuntimeError("=> no pretrained model found at '{}'".format(args.pretrained_model_path))

        # checkpoint = torch.load(args.pretrained_model_path)
        # Net.load_state_dict(checkpoint['state_dict_decomp'], strict=False)
        # Net2.load_state_dict(checkpoint['state_dict_enhance'], strict=False)
        # # args.start_epoch = checkpoint['epoch']
        # opt_Net.load_state_dict(checkpoint['optimizer_decomp'])
        # opt_Net2.load_state_dict(checkpoint['optimizer_enhance'])
        # best_pred = checkpoint['best_pred']

        # 只加载decomp
        checkpoint = torch.load(args.pretrained_model_path)
        Net.load_state_dict(checkpoint['state_dict_decomp'])
        opt_Net.load_state_dict(checkpoint['optimizer_decomp'])
        best_pred = 0.1

        print('加载decomp')
    else:
        # best_pred = 0.0
        # best_pred = 0.57565
        print('未加载lisu-pre')
        best_pred = 0.1

    trainset = LLRGBD_real(args, mode='train')
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    valset = LLRGBD_real(args, mode='val')
    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    # trainset = LLRGBD_synthetic(args, mode='train')
    # valset = LLRGBD_synthetic(args, mode='val')
    # == trainloader = DataLoader(trainset, batch_size=12, shuffle=True, num_workers=16, drop_last=True)
    # == valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=16, drop_last=False)

    if args.mode == 'train_and_val':
        train_and_val(args, Net, opt_Net, Net2, opt_Net2, trainloader, valloader, best_pred=best_pred)
    elif args.mode == 'output':
        output(args, Net, Net2, valloader)
    elif args.mode == 'evaluation':
        evaluation(args, Net, Net2, valloader)


if __name__ == '__main__':
    main(sys.argv[1:])
