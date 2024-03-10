# -*- coding: utf-8 -*-
# @Author  : ZhengChang
# @Email   : changzheng18@mails.ucas.ac.cn
# @Software: PyCharm
import torch
import torch.nn as nn
import math


class DiscriminatorContext(nn.Module):
    def __init__(self, height, width, in_channels, hidden_channels): #64,64,192,64
        super(DiscriminatorContext, self).__init__()

        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.n = int(math.log2(height))

        #全局判别器
        self.gd = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.hidden_channels,
                      kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, self.hidden_channels),
            nn.ReLU()
        )
        for i in range(self.n - 1):  #循环，直到输出为1*1*hidden_channels
            self.gd.add_module(name='conv_{0}'.format(i + 1),
                                 module=nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels,
                                                  kernel_size=3, stride=2, padding=1))
            self.gd.add_module(name='gn_{0}'.format(i + 1),
                                 module=nn.GroupNorm(4, self.hidden_channels))
            self.gd.add_module(name='relu_{0}'.format(i + 1),
                                 module=nn.ReLU())

        #局部判别器
        self.ld = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.hidden_channels,
                      kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, self.hidden_channels),
            nn.ReLU()
        )
        for i in range(self.n - 1):  # 循环，直到输出为1*1*hidden_channels
            self.ld.add_module(name='conv_{0}'.format(i + 1),
                                 module=nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels,
                                                  kernel_size=3, stride=2, padding=1))
            self.ld.add_module(name='gn_{0}'.format(i + 1),
                                 module=nn.GroupNorm(4, self.hidden_channels))
            self.ld.add_module(name='relu_{0}'.format(i + 1),
                                 module=nn.ReLU())


        self.linear_in_channels = int(math.ceil(float(width) / (2 ** self.n)) * self.hidden_channels * 2)  #ld和gd
        self.linear = nn.Sequential(
            nn.Linear(self.linear_in_channels, 1),
            nn.Sigmoid()
        )

       # print("Discriminator main Start")
       # print(self.main)
       # print(self.linear)
       # print("Discriminator main End")

    def forward(self, input_tensor):
        output_tensor = []
        output_gd_features = []
        output_ld_features = []
        for i in range(input_tensor.shape[1]): #input_tensor.shape[1]为图片数量，这里等于9

            #截取其中重点的区域，补零后，输入到ld
            ld_input_tensor = torch.zeros(input_tensor.shape)
            ld_input_tensor[:, :, :, 128:224, :] = input_tensor[:, :, :,128:224, :]
            ld_features =  self.ld(input_tensor[:, i, :])
            ld_features = ld_features.reshape([ld_features.shape[0], -1])  # 1，64

            #全局
            gd_features = self.gd(input_tensor[:, i, :])  #1,i, 64,64,192
            gd_features = gd_features.reshape([gd_features.shape[0], -1]) #1，64

            #链接, 1,128
            total_features = torch.cat([ld_features,gd_features], dim=-1)

            output_gd_features.append(gd_features)
            output_ld_features.append(ld_features)

            output = self.linear(total_features) #计算出值，单个浮点数
            output_tensor.append(output)

        output_tensor = torch.cat(output_tensor, dim=1)
        output_tensor = torch.mean(output_tensor, dim=1) #单个浮点数
        output_gd_features = torch.stack(output_gd_features, dim=1)#1，9，128
        output_ld_features = torch.stack(output_ld_features, dim=1)  # 1，9，128
        return output_tensor, output_gd_features, output_ld_features


