from utils import L2Norm, Detect
from utils.box import PriorBox
from configs import VOC

import torch.nn.functional as F
import torch.nn as nn
import torch
import os

base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],
    '512': [],
}


class SSD(nn.Module):
    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase  # 'train'或'test'
        self.size = size  # 输入尺寸=300
        self.num_classes = num_classes  # 类别数
        self.cfg = VOC  # 配置信息
        self.priorbox = PriorBox(self.cfg)  # 产生先验框
        with torch.no_grad():  # 不使用梯度
            self.priors = self.priorbox.forward()
        self.vgg = nn.ModuleList(base)  # 根据'base'建立nn.ModuleList对象
        self.L2Norm = L2Norm(512, 20)  # conv4_3后使用L2正则化
        self.extras = nn.ModuleList(extras)  # VGG-16后额外添加的四层

        self.loc = nn.ModuleList(head[0])  # 位置预测
        self.conf = nn.ModuleList(head[1])  # 置信度预测

        if phase == 'test':  # 测试阶段需要不同处理
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 200, 0.01, 0.45)

    def forward(self, x):
        sources = list()  # 用于存放分类层结果
        loc = list()  # 位置预测结果
        conf = list()  # 置信度预测结果

        for k in range(23):  # 获得VGG-16输出(到conv4_3截止)
            x = self.vgg[k](x)
        s = self.L2Norm(x)  # conv4_3后的正则化层
        sources.append(s)  # 分类层1

        for k in range(23, len(self.vgg)):  # conv6-conv7
            x = self.vgg[k](x)
        sources.append(x)  # 分类层2

        for k, v in enumerate(self.extras):  # 额外添加的4层
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)  # 分类层3、4、5、6

        # 根据sources的内容获得位置预测和置信度预测
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":  # 测试阶段需要不同处理
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # 位置预测
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),  # 置信度预测
                self.priors.type(type(x.data))  # 先验框
            )
        else:
            output = (  # 训练阶段
                loc.view(loc.size(0), -1, 4),  # 位置预测
                conf.view(conf.size(0), -1, self.num_classes),  # 置信度预测
                self.priors  # 先验框
            )
        # output[0].shape=(batch_size,8732,4)
        # output[1].shape=(batch_size,8732,num_classes)
        # output[2].shape=(8732,4)
        return output

    # 加载权重
    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def vgg(cfg, i, batch_norm=False):
    # cfg=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',512, 512, 512]
    layers = []
    in_channels = i  # 3
    for v in cfg:  # 原VGG-16种除全连接层部分
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    # VGG-16全连接层部分
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers  # 各层以列表形式返回


def add_extras(cfg, i, batch_norm=False):
    # cfg=[256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
    layers = []
    in_channels = i  # 1024
    flag = False
    for k, v in enumerate(cfg):  # 额外添加的四层
        if in_channels != 'S':
            if v == 'S':  # (1,3)[True]=3,(1,3)[False]=1
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers  # 各层以列表形式返回


def multibox(vgg, extra_layers, cfg, num_classes):
    # cfg=[4, 6, 6, 6, 4, 4]
    loc_layers = []  # 分类层
    conf_layers = []  # 回归层

    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):  # VGG部分两个预测层
        loc_layers += [nn.Conv2d(vgg[v].out_channels,  # 分类层
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,  # 回归层
                                  cfg[k] * num_classes, kernel_size=3, padding=1)]

    for k, v in enumerate(extra_layers[1::2], 2):  # 其他部分的预测层
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]  # 分类层
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]  # 回归层
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    '''
    首先，vgg函数的输入参数是配置信息[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 
    512, 512, 512, 'M', 512, 512, 512]和图像输入通道数3，以列表的形式返回VGG-16的前面几
    层(不包括conv6和conv7)；
    其次，add_extras函数的输入参数是配置信息[256, 'S', 512, 128, 'S', 256, 128, 256, 
    128, 256]和输入通道数1024，以列表形式返回VGG-16中的conv6、conv7以及额外添加的四个卷积
    层；  vgg, extra_layers, (loc_layers, conf_layers)
    然后，mobx[str(size)]=[4,6,6,6,4,4]表示特征图上每个位置的框数目；
    最后，multibox函数根据以上参数，以列表形式返回vgg、extra_layers，以及位置预测层和置信度
    预测层，即base_=vgg、extras_=extra_layers、head_=(loc_layers,conf_layers)。
    '''
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)
