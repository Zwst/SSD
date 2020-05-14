from data import VOCDetection, VOCAnnotationTransform
from configs import VOC_CLASSES, VOC_ROOT
from models.ssd import build_ssd

from matplotlib import pyplot as plt
from torch.autograd import Variable
import numpy as np
import torch
import cv2

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# 加载数据集
testSet = VOCDetection(VOC_ROOT, 'val', VOCAnnotationTransform())  # 测试数据集
img_id = 244
image = testSet.pull_image(img_id)  # 返回cv2格式
# 数据的预处理
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
x = cv2.resize(image, (300, 300)).astype(np.float32)
x -= (104.0, 117.0, 123.0)
x = x.astype(np.float32)
x = x[:, :, ::-1].copy()
x = torch.from_numpy(x).permute(2, 0, 1)
xx = Variable(x.unsqueeze(0))
if torch.cuda.is_available():
    xx = xx.cuda()

# 定义模型
net = build_ssd(phase='test', size=300, num_classes=21)  # 初始化模型
net.load_weights('D:/PyCharm/PyTorch/SSD/weights/ssd300_mAP_77.43_v2.pth')  # 载入权重
y = net(xx)  # 前向传播

# 解析并显示检测结果
plt.figure(figsize=(10, 10))
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
plt.imshow(image)
currentAxis = plt.gca()

detections = y.data  # detections.shape=(1,21,200,5)
scale = torch.Tensor(image.shape[1::-1]).repeat(2)  # 求图片宽高
for i in range(detections.size(1)):
    j = 0
    while detections[0, i, j, 0] >= 0.6:  # 只取置信度大于等于0.6的结果
        score = detections[0, i, j, 0]  # 置信度
        label_name = VOC_CLASSES[i - 1]  # 数字标签转换为符号标签
        display_txt = '%s: %.2f' % (label_name, score)  # 显示文本
        pt = (detections[0, i, j, 1:] * scale).cpu().numpy()  # 边界框坐标由相对结果转换为绝对结果
        coord = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1  # 坐标转换
        color = colors[i]  # 颜色
        currentAxis.add_patch(plt.Rectangle(*coord, fill=False, edgecolor=color, linewidth=2))  # 添加内容
        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.2})    # 添加文本
        j += 1

plt.show()
