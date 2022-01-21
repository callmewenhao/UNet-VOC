## UNet on VOC2007

### 数据集：VOC2007

- 一共 20 类物体 + 1 类背景
- 模型不考虑标签的边界 😎，这句代码 `segment[segment > self.num_classes] = 0` 将边界值变成了0。

### model：UNet

原论文中的 **UNet** 结构如下：

<img src="figures\UNET.png" style="zoom:60%;" />

实现的不同之处 👏：

- <font color=blue>conv3x3, ReLU </font>使用等宽卷积，保证了输入与输出图片的 shape 一致；
- <font color=green>up-conv 2x2</font> 使用 ConvTranspose2d，没有使用 **bilinear**；
- model 的具体实现注意查看 model.py

### loss

使用的多分类交叉熵：`nn.CrossEntropyLoss()`

交叉熵损失函数可以用在大多数语义分割场景中

- 二值交叉熵损失函数

- 多分类交叉熵损失函数

  注意pytorch中多分类交叉熵损失函数的输入形式！

另外，可以尝试使用 `Dice Loss`，公式如下：

<img src="figures\dice loss.png" style="zoom:70%;" />

为了使 S 取值范围是[0, 1]，分子中要乘 2 。

### 训练

**超参数：**

```
batch_size = 4
learning_rate = 1e-5
epochs = 50
```

**优化器：**`optim.Adam`

**损失函数：**`nn.CrossEntropyLoss()`

### 效果

> 整体效果一般般🤣

人的分割效果还行（下图依次为：原RGB图、标签图、预测图）

<img src="figures\img_seg_pred.png" style="zoom:50%;" />

其他物体的分割效果就不好了，只能找到大体轮廓，甚至类别都不对。

<img src="figures\img_seg_pred1.png" style="zoom:50%;" />

<img src="figures\img_seg_pred2.png" style="zoom:50%;" />

