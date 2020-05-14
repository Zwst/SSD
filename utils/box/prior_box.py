from itertools import product as product
from math import sqrt as sqrt
import torch


class PriorBox(object):
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']  # 300
        self.num_priors = len(cfg['aspect_ratios'])  # 6
        self.variance = cfg['variance'] or [0.1]  # 方差
        self.feature_maps = cfg['feature_maps']  # 特征图大小=[38,19,10,5,3,1]
        self.min_sizes = cfg['min_sizes']  # [30, 60, 111, 162, 213, 264]
        self.max_sizes = cfg['max_sizes']  # [60, 111, 162, 213, 264, 315]
        self.steps = cfg['steps']  # [8, 16, 32, 64, 100, 300]
        self.aspect_ratios = cfg['aspect_ratios']  # [[2],[2,3],[2,3],[2,3],[2],[2]]
        self.clip = cfg['clip']  # True
        self.version = cfg['name']  # VOC

        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []  # 存放每个特征的结果
        for k, f in enumerate(self.feature_maps):
            # product((0,1,2,...,f-1),(0,1,2,...,f-1))笛卡尔积，
            # 共f^2个结果, (i,j)为每个像素点的坐标
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]  # feature_map的大小
                cx = (j + 0.5) / f_k  # 根据公式求中心坐标
                cy = (i + 0.5) / f_k

                # 先验框比例为1
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # 先验框比例为1时的额外box，sqrt(s_k*s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # 其他的宽高比，根据k值确定是([2,3]还是[2])4个框还是2个框
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)

        return output   # output.shape(num_priors,4)
