from utils.box.box_utils import match, log_sum_exp
from configs import VOC as cfg

from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch


class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, neg_pos, use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes  # 类别数
        self.threshold = overlap_thresh  # 正负样本IoU阈值
        self.negpos_ratio = neg_pos  # 负正样本比例为3
        self.use_gpu = use_gpu  # 是否使用GPU

        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        # predictions=((batch_size,8732,4),(batch_size,8732,num_classes),(num_priors,4))
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)  # batch_size
        priors = priors[:loc_data.size(1), :]  # 先验框信息
        num_priors = (priors.size(0))  # 8732

        # 初始化变量用于存放标注框匹配的先验框的位置信息，shape=(batch_size,8732,4)
        loc_t = torch.Tensor(num, num_priors, 4)
        # 初始化变量用于存放标注框匹配的先验框的置信度信息，shape=(batch_size,8732)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):  # 遍历当前批次的图片
            # targets.shape=(batch_size,num_obj,5)
            truths = targets[idx][:, :-1].data  # 取位置标注信息，(num_obj,4)
            labels = targets[idx][:, -1].data  # 取类别标注信息，(num_obj,1)
            defaults = priors.data  # 先验框信息
            # 调用match函数完成标注框和先验框的匹配，结果存放在loc_t和conf_t中
            # loc_t.shape=(batch_size,num_priors,4)，conf_t.shape(batch_size,num_priors)
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0  # 去除背景标签0，即没有匹配标注框的样本
        # pos_idx=(batch_size,num_priors,4)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)

        loc_p = loc_data[pos_idx].view(-1, 4)  # 预测位置信息
        loc_t = loc_t[pos_idx].view(-1, 4)  # 先验位置信息
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)  # 位置损失

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # 计算softmax损失部分
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # 负样本难例挖掘
        loss_c = loss_c.view(pos.size()[0], pos.size()[1])  # 正样本对应损失值置为0
        loss_c[pos] = 0
        loss_c = loss_c.view(num, -1)  # reshape
        _, loss_idx = loss_c.sort(1, descending=True)  # 损失降序排序并返回索引
        _, idx_rank = loss_idx.sort(1)  # 索引升序排序，并返回相应索引
        num_pos = pos.long().sum(1, keepdim=True)  # 正样本数量
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)  # 负样本数量
        neg = idx_rank < num_neg.expand_as(idx_rank)  # 取损失值较大对应于前景预测很差、背景预测好的样本作为难例负样本

        pos_idx = pos.unsqueeze(2).expand_as(conf_data)  # 正样本索引
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)  # 负样本索引

        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)  # 置信度预测
        targets_weighted = conf_t[(pos + neg).gt(0)]  # 置信度标注
        # 计算第二部分损失函数
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)  # 置信度损失

        N = num_pos.data.sum()  # 损失标准化
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
