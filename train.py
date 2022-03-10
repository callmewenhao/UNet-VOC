import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import MyDataset
from model import UNET
from metrics import Evaluator
from torchvision.utils import save_image


# train one epoch
def training(model, data_loader, criterion, optimizer, evaluator, scheduler, device, epoch, epochs):
    model.train()
    losses = []
    for i, (image, segment) in enumerate(data_loader):

        image, segment = image.to(device, dtype=torch.float32), segment.to(device, dtype=torch.long)
        # 前向计算
        out = model(image)
        # 计算 loss
        loss = criterion(out, segment)
        losses.append(loss.item())
        # 反向传播，更新梯度
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # 在训练过程中验证
        evaluator.reset()
        pred = out.data.cpu().numpy()  # output.cpu().numpy() ?
        pred = np.argmax(pred, axis=1)
        gt = segment.cpu().numpy()
        evaluator.add_batch(gt, pred)
        acc_batch = evaluator.Pixel_Accuracy()

        if i % 50 == 0 and i > 0:
            print(f'epoch:{epoch}/{epochs}, iter:{i}th, train loss:{loss.item()}, AccPerBatch:{acc_batch}')

    mean_loss = sum(losses) / len(losses)
    print(f"loss at epoch {epoch} is {mean_loss}")
    scheduler.step(mean_loss)


# validation per some epochs
def validation(model, data_loader, evaluator, device, img_save_dir, epoch):
    model.eval()
    evaluator.reset()
    # test_loss = 0.0
    for i, (image, segment) in enumerate(data_loader):
        image, segment = image.to(device, dtype=torch.float32), segment.to(device, dtype=torch.long)
        output = model(image)
        # loss = criterion(output, segment)
        # test_loss += loss.item()
        pred = output.data.cpu().numpy()  # output.cpu().numpy() ?
        gt = segment.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        # Add batch sample into evaluator
        evaluator.add_batch(gt, pred)

        # if i % 10 == 0 and i > 0:
        #     # save some pred to check results
        #     _segment = segment[0]
        #     _output = output[0]  # output维度要改！
        #     img = torch.stack([_segment, _output], dim=0)
        #     save_image(img, f"{img_save_dir}\\{epoch}_{i}.png")

    # Fast test during the training
    Acc = evaluator.Pixel_Accuracy()
    # Acc_class = evaluator.Pixel_Accuracy_Class()
    # mIoU = evaluator.Mean_Intersection_over_Union()
    # FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    print(f"Pixel Accuracy: {Acc}")


def main():
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # path
    weight_save_dir = 'outputs'
    img_save_dir = 'outputs'
    data_path = 'F:\GithubRepository\图像分割数据集\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007'

    # params
    num_class = 21  # 21类
    batch_size = 4
    learning_rate = 1e-5
    epochs = 200

    # 数据集和模型
    dataset = MyDataset(data_path, num_class, (128, 128))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = UNET(3, num_class).to(device)  # 模型输入3channel，输出21channel

    # 先尝试加载之前的权重信息
    # if os.path.exists(weight_path):
    #     model.load_state_dict(torch.load(weight_path))
    #     print('load weights successfully!')
    # else:
    #     print('no weights!')
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5, verbose=True  # 5个epoch检查一次loss，触发后打印出lr
    )
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    evaluator = Evaluator(num_class)

    # training
    for epoch in range(epochs):
        training(model, data_loader, criterion, optimizer, evaluator, scheduler, device, epoch, epochs)
        if epoch % 10 == 0 and epoch > 0:  # validation
            validation(model, data_loader, evaluator, device, img_save_dir, epoch)

    # 训练结束后保存参数
    torch.save(model.state_dict(), f"{weight_save_dir}\\unet_{epochs}.pth")  # 每个epoch结束后都存一个参数


if __name__ == "__main__":
    main()


