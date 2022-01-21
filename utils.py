import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt


def keep_image_size(image_path, size=(256, 256)):
    """
    处理输入
    输入图片地址
    输出缩放后的彩色图片
    """
    img = Image.open(image_path)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    # print(mask)
    return mask

def keep_segment_size(image_path, size=(256, 256)):
    """
    处理标签
    输入图片地址
    输出缩放后的灰度图片
    """
    img = Image.open(image_path)
    temp = max(img.size)
    mask = Image.new('P', (temp, temp), 0)
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask


def main():
    test_image_path = "F:\GithubRepository\图像分割\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\SegmentationClass\\000033.png"
    segment = keep_segment_size(test_image_path, (128, 128))
    print(segment.size)
    data = np.array(segment)
    data[data > 21] = 0
    print(data.max(), data.min())
    label = torch.Tensor(data)
    print(label.shape)

if __name__ == "__main__":
    main()


