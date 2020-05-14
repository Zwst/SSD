from numpy import random
import numpy as np
import cv2


# ---     变换1     ---
# 将整形数值转换为浮点型
class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


# ---     变换2     ---
# 转换为绝对坐标
class ToAbsoluteCoord(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        return image, boxes, labels


# ---     变换3     ---
# 随机对比度
class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.lower >= 0, "contrast lower must be no-negative."
        assert self.upper >= self.lower, "contrast upper must be >= lower."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


# 变换颜色通道
class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


# 随机饱和度
class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.lower >= 0, "contrast lower must be no-negative."
        assert self.upper >= self.lower, "contrast upper must be >= lower."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
        return image, boxes, labels


# 随机明度
class RandomHue(object):
    def __init__(self, delta=18.0):
        assert 0.0 <= delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


# 随机亮度
class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


# 交换通道
class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image


# 随机噪声
class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            image = shuffle(image)
        return image, boxes, labels


# 图像失真
class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),  # 随机对比度
            ConvertColor(transform='HSV'),  # 转换颜色通道
            RandomSaturation(),  # 随机饱和度
            RandomHue(),  # 随机明度
            ConvertColor(current='HSV', transform='BGR'),  # 转换颜色通道
            RandomContrast()  # 随机对比度
        ]
        self.rand_brightness = RandomBrightness()  # 随机亮度
        self.rand_light_noise = RandomLightingNoise()  # 随机噪声

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


# ---     变换4     ---
# 随机图像拉伸
class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape

        ratio = random.uniform(1, 4)

        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top): int(top + height), int(left): int(left + width)] \
            = image

        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


# ---     变换5     ---

# 交集，box_a是一组box、box_b是一个box，下同
def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


# IoU
def iou(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union


# 随机图像裁剪
class RandomSampleCrop(object):
    def __init__(self):
        self.sample_options = (
            None,
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            (None, None)
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            model = random.choice(self.sample_options)
            # 随机到None直接返回
            if model is None:
                return image, boxes, labels
            # 最大IoU和最小IoU
            min_iou, max_iou = model
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            for _ in range(50):  # 迭代50次
                current_image = image
                # 宽和高随机采样
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # 宽高比例不当
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # 框坐标x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # 求iou
                overlap = iou(boxes, rect)
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # 裁剪图像
                current_image = current_image[rect[1]: rect[3], rect[0]: rect[2], :]

                # 中心点坐标
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                mask = m1 * m2
                if not mask.any():
                    continue

                current_boxes = boxes[mask, :].copy()

                current_labels = labels[mask]

                # 根据图像变换调整box
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                current_boxes[:, 2:] -= rect[:2]

                # 返回变换后的图像、box和label
                return current_image, current_boxes, current_labels


# ---     变换6     ---
# 随机图像镜像
class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


# ---     变换7     ---
# 转换为相对坐标
class ToPercentCoord(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height
        return image, boxes, labels


# ---     变换8     ---
# 固定图片尺寸(300,300)
class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels


# ---     变换9     ---
# 减均值
class SubtractMeans(object):
    def __init__(self, mead):
        self.mean = np.array(mead, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


# 将多种数据变换组合
class Compose(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transform:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


# 汇总变换
class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoord(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoord(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)
