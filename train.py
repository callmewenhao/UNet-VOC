import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import MyDataset
from model import UNET

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# path
weight_path = 'outputs\\unet.pth'
data_path = 'F:\GithubRepository\图像分割\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007'
img_save_path = 'outputs'


# params
batch_size = 4
learning_rate = 1e-5
epochs = 50


# train
def main():
    # 数据集和模型
    mydataset = MyDataset(data_path, 21, (128, 128))
    data_loader = DataLoader(mydataset, batch_size=batch_size, shuffle=True)
    model = UNET(3, 21).to(device)  # 模型输入3channel，输出21channel

    # 先尝试加载之前的权重信息
    # if os.path.exists(weight_path):
    #     model.load_state_dict(torch.load(weight_path))
    #     print('load weights successfully!')
    # else:
    #     print('no weights!')
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数

    for epoch in range(1, epochs+1):
        for i, (image, segment) in enumerate(data_loader):
            image, segment= image.to(device, dtype=torch.float32), segment.to(device, dtype=torch.long)
            # 前向计算
            out = model(image)
            # 计算 loss
            loss = criterion(out, segment)
            # 反向传播，更新梯度
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if i % 5 == 0:
                print(f'epoch:{epoch}/{epochs}, iter:{i}th, train loss:{loss.item()}')

            # _image = image[0]  # 查看第0张图片
            # _segment = segment[0]
            # _out = out[0]
            # img = torch.stack([_image, _segment, _out], dim=0)
            
            # save_image(img, f"{img_save_path}/{epoch}_{i}.png")
            
    torch.save(model.state_dict(), weight_path)  # 每个epoch结束后都存一个参数


if __name__ == "__main__":
    main()


