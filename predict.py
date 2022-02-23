import torch
from model import UNET
from dataset import transform
from utils import resizeImage, resizeMask, getPalette, tensorToPImage
import matplotlib.pyplot as plt


weight_path = 'outputs\\unet_200.pth'
test_image_path = "F:\GithubRepository\图像分割数据集\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages\\003889.jpg"
test_seg_path = "F:\GithubRepository\图像分割数据集\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\SegmentationClass\\003889.png"
seg = resizeMask(test_seg_path, (128, 128))

# 模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNET(3, 21).to(device)
model.load_state_dict(torch.load(weight_path))  # 加载参数

# 拿测试图片数据 预测
model.eval()
image = resizeImage(test_image_path, size=(128, 128))
data = transform(image).unsqueeze(dim=0).to(device)
pred = model(data)[0]

pred_seg = pred.cpu().detach().numpy().argmax(0)
p = getPalette(test_seg_path)
pred_seg = tensorToPImage(pred_seg, p)

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.subplot(1, 3, 2)
plt.imshow(seg)
plt.subplot(1, 3, 3)
plt.imshow(pred_seg)
plt.show()



