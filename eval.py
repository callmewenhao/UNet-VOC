import torch
import numpy as np
from model import UNET
from dataset import transform
from utils import keep_image_size, keep_segment_size
import matplotlib.pyplot as plt


weight_path = 'outputs\\unet.pth'
test_image_path = "F:\GithubRepository\图像分割\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages\\000068.jpg"
test_seg_path = "F:\GithubRepository\图像分割\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\SegmentationClass\\000068.png"
seg = keep_segment_size(test_seg_path, (128, 128)).convert('L')
seg = np.array(seg)
seg[seg > 21] = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNET(3, 21).to(device)
model.load_state_dict(torch.load(weight_path))  # 加载参数

# 拿一张测试图片 预测
model.eval()
image = keep_image_size(test_image_path, size=(128, 128))
data = transform(image).unsqueeze(dim=0).to(device)
pred = model(data)[0]
# print(pred.shape)

pred_seg = pred.cpu().detach().numpy().argmax(0)
print(seg.max(), pred_seg.max())

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.subplot(1, 3, 2)
plt.imshow(seg)
plt.subplot(1, 3, 3)
plt.imshow(pred_seg)
plt.show()



