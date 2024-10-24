# ResNet 网络详解

## 1. 什么是 ResNet ？

ResNet 网络是在 2015 年由微软实验室中的 何凯明 等几位大神提出，论文地址是[《Deep Residual Learning for Image Recognition》](https://arxiv.org/pdf/1512.03385.pdf)；是在 CVPR 2016 发表的一种影响深远的网络模型，由 何凯明 大神团队提出来，在 ImageNet 的分类比赛上将网络深度直接提高到了152层，前一年夺冠的 VGG 只有19层。斩获当年 ImageNet 竞赛中分类任务第一名，目标检测第一名。获得 COCO 数据集中目标检测第一名，图像分割第一名，可以说 ResNet 的出现对深度神经网络来说具有重大的历史意义。

## 2. ResNet 解决什么问题？

Resnet 利用跨层连接 (shortcut connection)，将输入信号直接添加到残差块的输出上，解决了深度网络中模型退化的问题 (degradation problem)。即更深的网络会伴随梯度消失/爆炸问题，从而阻碍网络的收敛。这种设计使得网络在反向传播时能够更容易地传递梯度，从而解决了深层网络训练中的梯度消失问题。

## 3. 解决办法

- 为了解决梯度消失或梯度爆炸问题，ResNet 论文提出通过数据的预处理以及在网络中使用 BN（Batch Normalization）层来解决。

- 为了解决深层网络中的退化问题，可以人为地让神经网络某些层跳过下一层神经元的连接，隔层相连，弱化每层之间的强联系。这种神经网络被称为残差网络 (ResNets)。ResNet 论文提出了 residual 结构（残差结构）来减轻退化问题，下图是使用 residual 结构的卷积网络，可以看到随着网络的不断加深，效果并没有变差，而是变的更好了。

（虚线是 train error，实线是 test error）

![residual 结构的比较](./images/ResNet-residual-comp.png)

## 4. ResNet 网络结构

### 4.1 基本结构

在 ResNet 中，令 H(x) = F(x) + x ，如下图：

![ResNet 网络基本单元](./images/ResNet-base-struction.jpg)

残差模块：一条线路不变（恒等映射 x）；另一条线路负责拟合相对于原始网络的残差 F(x)，去纠正原始网络的偏差，而不是让整体网络去拟合全部的底层映射，这样网络只需要纠正偏差。

### 4.2 两种 Block

ResNet block 有两种，一种左侧两层的 BasicBlock 结构，一种是右侧三层的 bottleneck 结构，如下图：

![两种 ResNet Block 的比较](./images/ResNet-two-block-comp.png)

`bottleneck` 结构的优点是：既保持了模型精度又减少了网络参数和计算量，节省了计算时间。

注意：一般，浅层网络用 BasicBlock 结构；深层网络，采用三层的 bottleneck 残差结构。

### 4.3 实线残差结构

![实线残差模块](./images/ResNet-solid-ResNet34.png) &nbsp; ![实线残差模块](./images/ResNet-solid-ResNet101.png)

### 4.4 虚线残差结构

![虚线残差结构](./images/ResNet-dashed-Reset34.png) &nbsp; ![虚线残差结构](./images/ResNet-dashed-Reset101.png)

区别：

- 实线残差结构的输入、输出特征矩阵维度是一样的，故可以直接进行相加。

- 虚线残差结构（conv3_x、conv4_x、conv5_x 第一层）将图像的高、宽和深度都改变了。

### 4.5 网络框架结构比较

![三种网络框架结构比较](./images/ResNet-struction-compare.jpg)

ResNet 主要有五种主要形式：Res18，Res34，Res50，Res101，Res152；

## 5. ResNet 模型完整代码

![文件目录结构](./images/ResNet-file-struction.png)

model.py

```python
"""
# 搭建resnet-layer模型
#
"""
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """搭建BasicBlock模块"""
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # 使用BN层是不需要使用bias的，bias最后会抵消掉
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)    # BN层, BN层放在conv层和relu层中间使用
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    # 前向传播
    def forward(self, X):
        identity = X
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.downsample is not None:    # 保证原始输入X的size与主分支卷积后的输出size叠加时维度相同
            identity = self.downsample(X)

        return self.relu(Y + identity)

class BottleNeck(nn.Module):
    """搭建BottleNeck模块"""
    # BottleNeck模块最终输出out_channel是Residual模块输入in_channel的size的4倍(Residual模块输入为64)，shortcut分支in_channel
    # 为Residual的输入64，因此需要在shortcut分支上将Residual模块的in_channel扩张4倍，使之与原始输入图片X的size一致
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        # 默认原始输入为224，经过7x7层和3x3层之后BottleNeck的输入降至64
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)    # BN层, BN层放在conv层和relu层中间使用
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)  # Residual中第三层out_channel扩张到in_channel的4倍

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    # 前向传播
    def forward(self, X):
        identity = X

        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))

        if self.downsample is not None:    # 保证原始输入X的size与主分支卷积后的输出size叠加时维度相同
            identity = self.downsample(X)

        return self.relu(Y + identity)

class ResNet(nn.Module):
    """搭建ResNet-layer通用框架"""
    # num_classes是训练集的分类个数，include_top是在ResNet的基础上搭建更加复杂的网络时用到，此处用不到
    def __init__(self, residual, num_residuals, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()

        self.out_channel = 64    # 输出通道数(即卷积核个数)，会生成与设定的输出通道数相同的卷积核个数
        self.include_top = include_top

        self.conv1 = nn.Conv2d(3, self.out_channel, kernel_size=7, stride=2, padding=3,
                               bias=False)    # 3表示输入特征图像的RGB通道数为3，即图片数据的输入通道为3
        self.bn1 = nn.BatchNorm2d(self.out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self.residual_block(residual, 64, num_residuals[0])
        self.conv3 = self.residual_block(residual, 128, num_residuals[1], stride=2)
        self.conv4 = self.residual_block(residual, 256, num_residuals[2], stride=2)
        self.conv5 = self.residual_block(residual, 512, num_residuals[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))    # output_size = (1, 1)
            self.fc = nn.Linear(512 * residual.expansion, num_classes)

        # 对conv层进行初始化操作
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def residual_block(self, residual, channel, num_residuals, stride=1):
        downsample = None

        # 用在每个conv_x组块的第一层的shortcut分支上，此时上个conv_x输出out_channel与本conv_x所要求的输入in_channel通道数不同，
        # 所以用downsample调整进行升维，使输出out_channel调整到本conv_x后续处理所要求的维度。
        # 同时stride=2进行下采样减小尺寸size，(注：conv2时没有进行下采样，conv3-5进行下采样，size=56、28、14、7)。
        if stride != 1 or self.out_channel != channel * residual.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.out_channel, channel * residual.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * residual.expansion))

        block = []    # block列表保存某个conv_x组块里for循环生成的所有层
        # 添加每一个conv_x组块里的第一层，第一层决定此组块是否需要下采样(后续层不需要)
        block.append(residual(self.out_channel, channel, downsample=downsample, stride=stride))
        self.out_channel = channel * residual.expansion    # 输出通道out_channel扩张

        for _ in range(1, num_residuals):
            block.append(residual(self.out_channel, channel))

        # 非关键字参数的特征是一个星号*加上参数名，比如*number，定义后，number可以接收任意数量的参数，并将它们储存在一个tuple中
        return nn.Sequential(*block)

    # 前向传播
    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.maxpool(Y)
        Y = self.conv5(self.conv4(self.conv3(self.conv2(Y))))

        if self.include_top:
            Y = self.avgpool(Y)
            Y = torch.flatten(Y, 1)
            Y = self.fc(Y)
        return Y

# 构建 ResNet-34 模型
def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


# 构建 ResNet-50 模型
def resnet50(num_classes=1000, include_top=True):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

# 模型网络结构可视化
net = resnet34()

"""
# 1. 使用torchsummary中的summary查看模型的输入输出形状、顺序结构，网络参数量，网络模型大小等信息
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = net.to(device)
summary(model, (3, 224, 224))    # 3是RGB通道数，即表示输入224 * 224的3通道的数据
"""

"""
# 2. 使用torchviz中的make_dot生成模型的网络结构，pdf图包括计算路径、网络各层的权重、偏移量
from torchviz import make_dot
X = torch.rand(size=(1, 3, 224, 224))    # 3是RGB通道数，即表示输入224 * 224的3通道的数据
Y = net(X)
vise = make_dot(Y, params=dict(net.named_parameters()))
vise.view()
"""

"""
# Pytorch官方ResNet模型
from torchvision.models import resnet34
"""

```

## x. 参考文章

- [【深度学习】ResNet网络讲解](https://blog.csdn.net/weixin_44001371/article/details/134192776)

- [跟着问题学7——ResNet网络详解及代码实战](https://zhuanlan.zhihu.com/p/716390397)

- [深入浅出之Resnet网络](https://blog.csdn.net/a8039974/article/details/142202414)
