import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt


def getPalette(img_path):
    """
    获取图片调色板信息
    """
    img = Image.open(img_path)
    palette = img.getpalette()
    return palette


def tensorToPImage(mask_np, palette):
    """
    输入：图片的nparray 和 调色板信息
    输出：对应位图
    """
    img = np.uint8(mask_np)
    img = Image.fromarray(img)
    img.putpalette(palette)
    return img


def resizeImage(image_path, size=(256, 256)):
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
    return mask


def resizeMask(image_path, size=(256, 256)):
    """
    处理标签
    输入位图片地址
    输出缩放后的位图
    """
    img = Image.open(image_path)
    p = img.getpalette()
    temp = max(img.size)
    mask = Image.new('P', (temp, temp), 0)
    mask.putpalette(p)
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask


# 测试代码
def main():
    test_image_path1 = "F:\GithubRepository\图像分割数据集\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\SegmentationClass\\000033.png"
    img1 = Image.open(test_image_path1)
    palette1 = img1.getpalette()
    img2 = np.array(img1)
    img3 = Image.fromarray(img2)
    img3.putpalette(palette1)
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.imshow(img1)
    plt.subplot(3, 1, 2)
    plt.imshow(img2)
    plt.subplot(3, 1, 3)
    plt.imshow(img3)
    plt.show()


if __name__ == "__main__":
    main()


