"""
@FileName: models.py
@Time    : 7/16/2020
@Author  : Ning Zhang
@GitHub: https://github.com/noahzn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LISU_DECOMP(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv_5 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1,
                                         output_padding=1)

        self.conv_6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_8 = nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1, stride=1, padding=0)

    def forward(self, img):
        c1 = self.conv_1(img)
        c2 = F.relu(self.conv_2(c1))
        c3 = F.relu(self.conv_3(c2))
        c4 = F.relu(self.conv_4(c3))

        c5 = F.relu(self.conv_5(c4))
        c6 = F.relu(self.conv_6(torch.cat((c2, c5), dim=1)))
        c7 = self.conv_7(torch.cat((c1, c6), dim=1))
        c8 = self.conv_8(c7)
        c9 = torch.sigmoid(c8)
        reflectance, illumination = c9[:, :3, :, :], c9[:, 3:4, :, :]

        return illumination, reflectance


class LISU_JOINT(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 4
        self.channels = [32, 64, 128, 256, 512]

        # Encoder layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.channels[0], 3, 1, 1),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.channels[0], self.channels[1], 3, 2, 1),
            nn.BatchNorm2d(self.channels[1]),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.channels[1], self.channels[2], 3, 2, 1),
            nn.BatchNorm2d(self.channels[2]),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(self.channels[2], self.channels[3], 3, 2, 1),
            nn.BatchNorm2d(self.channels[3]),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(self.channels[3], self.channels[4], 3, 2, 1),
            nn.BatchNorm2d(self.channels[4]),
            nn.ReLU()
        )

        # decoder
        self.mid_s = nn.Sequential(
            nn.Conv2d(self.channels[4], self.channels[4], 3, 1, 1),
            nn.BatchNorm2d(self.channels[4]),
            nn.ReLU(),
            nn.Conv2d(self.channels[4], self.channels[4], 3, 1, 1),
            nn.BatchNorm2d(self.channels[4]),
            nn.ReLU(),
            nn.Conv2d(self.channels[4], self.channels[4], 3, 1, 1),
            nn.BatchNorm2d(self.channels[4]),
            nn.ReLU(),
            nn.Conv2d(self.channels[4], self.channels[4], 3, 1, 1),
            nn.BatchNorm2d(self.channels[4]),
            nn.ReLU(), )

        self.deconv4_s = nn.Sequential(
            nn.Conv2d(self.channels[4]*2, self.channels[3], 3, 1, 1),
            nn.BatchNorm2d(self.channels[3]),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

        self.deconv3_s = nn.Sequential(
            nn.Conv2d(self.channels[3]*2, self.channels[2], 3, 1, 1),
            nn.BatchNorm2d(self.channels[2]),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

        self.deconv2_s = nn.Sequential(
            nn.Conv2d(self.channels[2]*2, self.channels[1], 3, 1, 1),
            nn.BatchNorm2d(self.channels[1]),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

        self.deconv1_s = nn.Sequential(
            nn.Conv2d(self.channels[1]*2, self.channels[0], 3, 1, 1),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), )

        self.out_s = nn.Sequential(
            nn.Conv2d(self.channels[0]*2, self.channels[0], 3, 1, 1),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(),
            nn.Conv2d(self.channels[0], 14, 3, 1, 1),
        )

    def forward(self, x):
        # ==encoder 1-5层
        x1 = self.conv1(x)
        # 16 x 240 x 320  # 32
        x2 = self.conv2(x1)
        # 32 x 120 x 160  # 64
        x3 = self.conv3(x2)
        # 64 x 60 x 80  # 128
        x4 = self.conv4(x3)
        # 128 x 30 x 40  # 256
        x5 = self.conv5(x4)
        # 256 x 15 x 20

        # #中间4块，最后一块远连接
        xmid_s = torch.cat((self.mid_s(x5), x5), dim=1)
        # 512 x 15 x 20

        x4d_s = self.deconv4_s(xmid_s)  # 256

        # #a添加mask分割？
        x3d_s = self.deconv3_s(torch.cat([x4d_s] + [x4], dim=1))

        x2d_s = self.deconv2_s(torch.cat([x3d_s] + [x3], dim=1))

        x1d_s = self.deconv1_s(torch.cat([x2d_s] + [x2], dim=1))

        out_s = self.out_s(torch.cat([x1d_s] + [x1], dim=1))

        return out_s, x1, x2