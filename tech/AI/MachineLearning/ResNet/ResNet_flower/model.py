#
# From: https://blog.csdn.net/weixin_44001371/article/details/134192776
#
import torch
import torch.nn as nn

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

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

# 构建 ResNet-18 模型
def resnet34(num_classes=1000, include_top=True, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top, **kwargs)

# 构建 ResNet-34 模型
def resnet34(num_classes=1000, include_top=True, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top, **kwargs)

# 构建 ResNet-50 模型
def resnet50(num_classes=1000, include_top=True, **kwargs):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top, **kwargs)

# 构建 ResNet-101 模型
def resnet101(num_classes=1000, include_top=True, **kwargs):
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top, **kwargs)

# 构建 ResNet-152 模型
def resnet152(num_classes=1000, include_top=True, **kwargs):
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top, **kwargs)

# 构建 ResNeXt-50 32x4d 模型
def resnext50_32x4d(num_classes=1000, include_top=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top, **kwargs)

# 构建 ResNeXt-101 32x8d 模型
def resnext101_32x8d(num_classes=1000, include_top=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top, **kwargs)

# 构建 Wide ResNet-50-2 模型
def wide_resnet50_2(num_classes=1000, include_top=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top, **kwargs)

# 构建 Wide ResNet-100-2 模型
def wide_resnet101_2(num_classes=1000, include_top=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top, **kwargs)

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