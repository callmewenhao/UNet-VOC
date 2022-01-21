import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from utils import keep_image_size, keep_segment_size
from torchvision.utils import save_image

transform = transforms.Compose([
    transforms.ToTensor(),  # convert a PIL image to tensor (H, W, C) in range [0,255] to a torch.Tensor(C, H, W) in the range [0.0, 1.0]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class MyDataset(Dataset):
    def __init__(self, path, num_classes=21, image_shape=(256, 256)):
        super().__init__()
        self.path = path  # 标签所在文件夹路径
        self.num_classes = num_classes
        self.image_shape = image_shape
        self.names = os.listdir(os.path.join(path, 'SegmentationClass'))

    def __getitem__(self, index):
        segment_name = self.names[index]
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
        image_path = os.path.join(self.path, 'JPEGImages', segment_name.replace('png', 'jpg'))
        image = keep_image_size(image_path, size=self.image_shape)
        segment = keep_segment_size(segment_path, size=self.image_shape)
        segment = np.array(segment)
        segment[segment > self.num_classes] = 0  # 边框处理
        return transform(image), torch.Tensor(segment)
        
    def __len__(self):
        return len(self.names)


def main():
    train_data_path = "F:\GithubRepository\图像分割\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007"
    mydataset = MyDataset(train_data_path, 21, (128, 128))
    _image = mydataset[0][0]
    _mask = mydataset[15][1]
    print(_image.shape, _mask.shape)
    print(_image.dtype, _mask.dtype)

if __name__ == "__main__":
    main()

