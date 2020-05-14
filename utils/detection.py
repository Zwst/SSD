from utils.box.box_utils import decode, nms
from configs import VOC as cfg

from torch.autograd import Function
import torch


class Detect(Function):
    # 测试阶段的SSD的最后层，用于解码预测信息，包括使用NMS及阈值筛选出top_k个预测结果
    def __init__(self, num_classes, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes  # 类别数
        self.top_k = top_k  # 保留结果数
        self.conf_thresh = conf_thresh  # 阈值
        self.nms_thresh = nms_thresh  # NMS阈值

        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.variance = cfg['variance']

    def forward(self, loc_data, conf_data, prior_data):
        # loc_data=(batch_size,num_priors,4)位置预测信息
        # conf_data=(batch_size,num_priors,num_classes)置信度预测信息
        # prior_data=(num_priors,4)先验框信息
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)  # 先验框数目
        # 初始化返回变量，shape=(batch_size,num_classes,top_k,5)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        # 置信度预测，shape=(batch_size,num_classes,num_priors)
        conf_preds = conf_data.view(num, num_priors, self.num_classes) \
            .transpose(2, 1)
        # 将预测结果解码为边界框信息
        for i in range(num):
            # decode函数将相对坐标转化为绝对坐标
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):  # 背景类为0
                # 返回由0和1组成的数组，0表示小于conf_thresh，1表示大于0表示小于conf_thresh
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                # 返回1对应的元素
                scores = conf_scores[cl][c_mask]
                # 没有框的情况
                if scores.size(0) == 0:
                    continue
                # 获得对应box的二值矩阵
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                # reshape
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # 使用NMS
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                # 拼接结果
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        # 将结果降序排序并取top_k个结果
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        # output.shape=(batch_size,num_classes,top_k,5)
        return output
