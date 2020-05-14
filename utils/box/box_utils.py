import torch


def point_form(boxes):
    # 将boxes转化为(xmin,ymin,xmax,ymax)形式
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)


def center_size(boxes):
    # 将boxes转化为(x,y,w,h)形式
    return torch.cat(((boxes[:, 2:] + boxes[:, :2]) / 2,
                      boxes[:, 2:] - boxes[:, :2]), 1)


def intersect(box_a, box_b):
    """
    求交集。
    :param box_a: 标注框 shape=(num_obj,4)
    :param box_b: 先验框 shape=(num_priors,4)
    由于二者的形状不一致，为了求标注框与先验框的交并比，我们首先需要将二者的形状统一。
    具体地，首先将二者的形状统一为(num_obj,num_priors,4)，然后再计算交集。由交集
    的定义，首先计算相交矩形的左上角位置；再计算相交矩形的右下角位置。位置的形状通过
    expand()函数扩展至(num_obj,num_priors,4)，然后求积。
    """
    # 分别求num_obj和num_priors
    a = box_a.size(0)
    b = box_b.size(0)
    # 分别求左上角位置和右下角位置
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(a, b, 2),
                       box_b[:, 2:].unsqueeze(0).expand(a, b, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(a, b, 2),
                       box_b[:, :2].unsqueeze(0).expand(a, b, 2))
    # 保证矩形存在
    inter = torch.clamp((max_xy - min_xy), min=0)
    # 求相交矩形区域面积，返回的形状为(a,b)，表示a个标注框和b个先验框对应的交集大小
    return inter[:, :, 0] * inter[:, :, 1]


def iou(box_a, box_b):
    """
    求交并比。
    :param box_a: 标注框，shape=(num_obj,4)
    :param box_b: 先验框，shape=(num_priors,4)
    根据交并比的定义，我们通过上述函数求得标注框与先验框的交集，然后求二者的并集，最后
    求得二者的交并比。在求交并比的过程中，同样需要将二者各自的形状统一至交集的形状，即
    (num_obj,num_priors)。
    """
    # 交集
    inter = intersect(box_a, box_b)
    # 各自的面积
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    # 并集
    union = area_a + area_b - inter
    # 求交并比，shape=(num_obj,num_priors)
    return inter / union


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    overlaps = iou(truths, point_form(priors))  # 标注框与先验框的IoU，shape=(num_obj,num_priors)
    # 对于每个标注框，取与其有最高IoU的先验框，并返回IoU和对应先验框的索引，shape=(1,num_obj)
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=False)
    # 对于每个先验框，取与其有最高IoU的标注框，并返回IoU和对应标注框的索引，shape=(1,num_priors)
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=False)
    # index_fill_函数的功能是在指定维度轴(0)上根据索引(best_prior_idx)使用值(2)填充
    # 具体到本语句，第一步找到与标注框有最大IoU的先验框索引best_prior_idx，第二步找到与先验框有最大IoU的
    # 标注框的IoU值，调用函数将与标注框有最高IoU的先验框之间的IoU值设置为2，即该标注框只与该先验框匹配，确
    # 定该先验框为正样本。
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    # 这里为了防止某个标注框没有先验框与之匹配，即两个或多个标注框与同一个先验框有最大IoU值。如果不存在标注
    # 框没有匹配对象的情况，下面语句执行前后不作任何改变；否则，将重复匹配对象的标注框给予强制匹配对象j。
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    # 匹配结果，shape=(num_priors,4)，即每个标注框所匹配的先验框的坐标
    matches = truths[best_truth_idx]
    # 置信度即IoU匹配情况，shape=(num_priors)
    conf = labels[best_truth_idx] + 1
    # 小于设定阈值的类置为背景
    conf[best_truth_overlap < threshold] = 0
    # 获得编码结果
    loc = encode(matches, priors, variances)
    # 给当前索引为idx的图像赋值匹配结果
    loc_t[idx] = loc
    conf_t[idx] = conf


def encode(matched, priors, variances):
    """
    :param matched: 与标注框匹配的先验框的坐标，shape=(num_priors,4)
    :param priors: 先验框的坐标，shape=(num_priors,4)
    """
    # 参考原文公式(2)，得到与中心坐标的偏移。
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    # 参考原文公式(2)，偏移值分别除以d_w、d_h。
    g_cxcy /= (variances[0] * priors[:, 2:])
    # 参考原文公式(2)，得到与宽高的偏移。
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    # 参考原文公式(2)，偏移值取对数。
    g_wh = torch.log(g_wh) / variances[1]
    # 将中心偏移与宽高偏移拼接，得到(num_priors,4)。
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


def decode(loc, priors, variances):
    """
    将偏移信息解码为图像上的实际坐标。
    :param loc: 坐标信息。
    :param priors: 先验信息。
    """
    # 根据公式(2)或encode()函数逆推解码即可。
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    # 计算LogSumExp(x)
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


def nms(boxes, scores, overlap=0.5, top_k=200):
    """
    :param boxes: 边界框，shape=(nums,4)。
    :param scores: 边界框对应的置信度，shape=(nums,1)。
    :param overlap: NMS阈值。
    :param top_k: 只取置信度前200的边界框。
    :return: 保留的边界框及数目。
    此函数里面涉及大量Python里面的数据切片，可以首先理解torch.index_select()、
    le()、torch.sort()、torch.clamp()等函数。
    """
    # 定义等长的变量keep用于存放保留的边界框索引。
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    # 取坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    # 按列从小到大排序，idx存放着排序后数据的原始索引。
    _, idx = scores.sort(0)
    # 取置信度最大的top_k个边界框
    idx = idx[-top_k:]
    # 初始化
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()
    count = 0
    while idx.numel() > 0:
        # 取置信度最大的边界框的索引i。
        i = idx[-1]
        # 加入结果
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        # 去掉最后一个元素，即去掉上述的最大置信度边界框对应的索引。
        idx = idx[:-1]
        # index_select()函数在指定维对数据切片。
        # 取x的的idx行，即保留符合条件的边界框的坐标，结果存放在out里。
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # i表示置信度最大的边界框的置信度，求其余框与索引为i的框的交集区域左上角坐标和右下角坐标。
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        # 求宽高
        w = xx2 - xx1
        h = yy2 - yy1
        # 检查w和h是否合法
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        # 交集
        inter = w * h
        # 剩余框的面积
        rem_areas = torch.index_select(area, 0, idx)
        # 并集
        union = rem_areas + area[i] - inter
        # iou
        IoU = inter / union
        # 仅保留iou小于设定的NMS阈值的边界框的索引。
        idx = idx[IoU.le(overlap)]
    # 返回保留的边界框的索引及其数量。
    return keep, count
