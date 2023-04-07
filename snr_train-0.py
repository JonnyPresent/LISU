import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.tarin_config import get_arguments
from dataset import LLRGBD_real

def save_model(args):
    if not os.path.isdir(args.save_model_path):
        os.makedirs(args.save_model_path)
    if miou > max_score:
        max_score = miou
        ckpt_path = os.path.join(self.args.save_model_path,
                                 'lisu-fifo_maxscore{1:.4f}_epoch{2}_{0}.pth'.format(epoch, max_score,
                                                                                     self.args.num_epochs))
        save_state = {'model': self.model_resnet.state_dict(),
                      'optimizer_enhance': self.opt_resnet.state_dict(),
                      'fogpass1_state_dict': self.FogPassFilter1.state_dict(),
                      'fogpass1_opt_state_dict': self.FogPassFilter1_optimizer.state_dict(),
                      'fogpass2_state_dict': self.FogPassFilter2.state_dict(),
                      'fogpass2_opt_state_dict': self.FogPassFilter2_optimizer.state_dict(),
                      'epoch': epoch}
        torch.save(save_state, ckpt_path)
        print('保存best:')
    elif epoch > 0 and epoch % 50 == 0:
        ckpt_path = os.path.join(self.args.save_model_path,
                                 'epoch{0}.pth'.format(epoch, max_score))
        save_state = {'model': self.model_resnet.state_dict(),
                      'optimizer_enhance': self.opt_resnet.state_dict(),
                      'fogpass1_state_dict': self.FogPassFilter1.state_dict(),
                      'fogpass1_opt_state_dict': self.FogPassFilter1_optimizer.state_dict(),
                      'fogpass2_state_dict': self.FogPassFilter2.state_dict(),
                      'fogpass2_opt_state_dict': self.FogPassFilter2_optimizer.state_dict(),
                      'epoch': epoch}
        torch.save(save_state, ckpt_path)
        print('保存断点')


def train_epoch(epoch, train_loader):
    model_decomp.eval()

    tbar = tqdm(train_loader)
    loss_list = []
    fpf_loss, loss, loss_r, loss_s = 0, 0, 0, 0
    for i, (image, image2, label, name) in enumerate(tbar):  # l lowlight highlight lable
        image = image.cuda()
        image2 = image2.cuda()
        label = label.cuda()
        # image_size = np.array(image.shape)

        I, R = self.model_snr(image)
        I2, R2 = self.model_snr(image2)
    pass


def train(args, train_loader, eval_loader):
    max_score = args.best_score
    for epoch in range(args.start_epoch, args.num_epochs):
        train_epoch(epoch, train_loader)
        acc, macc, miou = evaluate(eval_loader, epoch)

        # 保存模型



if __name__ == '__main__':
    args = get_arguments(sys.argv[1:])
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    print('cuda_device', os.environ["CUDA_VISIBLE_DEVICES"])
    print('start_epoch', args.start_epoch)

    trainset = LLRGBD_real(args, mode='train')
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    valset = LLRGBD_real(args, mode='val')
    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)