from configs.config import VOC, MEANS, VOC_ROOT
from data.augmentations import SSDAugmentation
from utils.loss import MultiBoxLoss
from models.ssd import build_ssd
from data import *

import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn.init as init
import torch.optim as optim
import torch.nn as nn
import argparse
import torch
import time
import os


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC',
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=4, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=1, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=True, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='D:/PyCharm/PyTorch/SSD/weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.visdom:
    import visdom

    viz = visdom.Visdom()

cfg = VOC


def train():
    # 构造数据集
    dataset = VOCDetection(root=args.dataset_root,
                           phase='train',
                           transform=SSDAugmentation(cfg['min_dim'], MEANS))
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # 建立SSD模型
    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:  # 初始权重
        print('Initializing weights...')
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)
    # 优化器
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    # 返回MultiBoxLoss类对象
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, 3, args.cuda)

    net.train()

    print('Loading the dataset...')
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)

    batch_iterator = iter(data_loader)  # 创建迭代器
    # 0-120000,迭代一次遍历batch_size张图像，共2501张训练图片，共需要2501//4=425批
    # 总共遍历图像的次数为120000/425=282次
    for iteration in range(args.start_iter, cfg['max_iter']):
        if iteration in cfg['lr_steps']:  # 调整学习率
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        try:  # 以迭代方式获取数据
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            with torch.no_grad():
                targets = [Variable(ann.cuda()) for ann in targets]
        else:
            images = Variable(images)
            with torch.no_grad():
                targets = [Variable(ann) for ann in targets]

        t0 = time.time()

        out = net(images)  # 前向传播
        optimizer.zero_grad()  # 梯度清零
        loss_l, loss_c = criterion(out, targets)  # 获得回归损失和分类损失
        loss = loss_l + loss_c  # 总损失值
        loss.backward()  # 反向传播
        optimizer.step()  # 参数优化

        t1 = time.time()
        # 打印信息
        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter: ' + repr(iteration) + ' || Loc_Loss: %.6f || Conf_Loss: %.6f || Total_Loss: %.6f ||'
                  % (loss_l.data.item(), loss_c.data.item(), loss.data.item()), end=' ')
        # 更新visdom显示
        if args.visdom:
            update_vis_plot(iteration, loss_l.data.item(), loss_c.data.item(),
                            iter_plot, 'append')
        # 打印信息
        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_VOC2007_' +
                       repr(iteration) + '.pth')
    # 保存模型
    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    # 在step个步骤后调整学习率
    lr = args.lr * (gamma ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    # xavier初始化方法
    init.xavier_uniform_(param)


def weights_init(m):
    # 权重初始化
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


# 创建visdom画图窗口
def create_vis_plot(xlabel, ylabel, title, legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=xlabel,  # x坐标
            ylabel=ylabel,  # y坐标
            title=title,  # 标题
            legend=legend  # 图注
        )
    )


def update_vis_plot(iteration, loc, conf, window1, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,  # 根据迭代次数绘制横坐标
        # 绘制做标标的三个损失值
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )


if __name__ == '__main__':
    train()
