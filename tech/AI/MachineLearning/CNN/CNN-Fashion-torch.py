#
# From: https://blog.csdn.net/qq_41813454/article/details/135992586
#
import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
# from d2l import torch as d2l
from matplotlib import pyplot as plt
import numpy as np
import sys

# -----------------------------获取并读取FashionMNIST数据集函数，返回小批量train，test-----------------------------------
def load_data_fashion_mnist(batch_size, image_size, root='~/.datasets/FashionMNIST'):
    data_transform = transforms.Compose([
        # transforms.ToPILImage(),
        # 这一步取决于后续的数据读取方式，如果使用内置数据集读取方式则不需要
        # transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    mnist_train = torchvision.datasets.FashionMNIST(root=root,
                                                    train=True, download=True, transform=data_transform)
    mnist_test  = torchvision.datasets.FashionMNIST(root=root,
                                                    train=False, download=True, transform=data_transform)

    # targets表示图像对应的类别标签（0-9）
    # mnist_train = TensorDataset(torch.tensor(mnist_train.data, dtype=torch.float), mnist_train.targets)
    # targets表示图像对应的类别标签（0-9）
    # mnist_test  = TensorDataset(torch.tensor(mnist_test.data, dtype=torch.float), mnist_test.targets)
    '''
    上面的 mnist_train,mnist_test 都是 torch.utils.data.Dataset 的子类，所以可以使用len()获取数据集的大小
    训练集和测试集中的每个类别的图像数分别是6000，1000，两个数据集分别有10个类别
    '''
    # mnist 是 torch.utils.data.dataset 的子类，因此可以将其传入torch.utils.data.DataLoader来创建一个DataLoader实例来读取数据
    # 在实践中，数据读取一般是训练的性能瓶颈，特别是模型较简单或者计算硬件性能比较高的时候
    # DataLoader一个很有用的功能就是允许多进程来加速读取，使用num_works来设置4个进程读取数据
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter  = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter

# 1. 获取训练集和测试集的DataLoader迭代器
image_size = 28
batch_size = 64

train_iter, test_iter = load_data_fashion_mnist(batch_size, image_size)

# image, label = next(iter(train_iter))
# print(image.shape, label.shape)

# plt.imshow(image[0][0], cmap="gray")

# 2. 定义网络模型
num_inputs = 784    # 由于softmax回归输入要求是向量，所以对于28×28的图片，应该拉长成784的向量作为输入
num_outputs = 10    # 10分类问题，输出为长度为10的向量，里面记录样本在各个类别上的预测概率

# 设置训练轮数为10轮
num_epochs = 10
# 学习率
lr = 0.001

# 3. 定义模型结构

#
# BugFix: ValueError: Expected input batch_size (256) to match target batch_size (32)
# See: https://www.cnblogs.com/zuiyixin/p/15573217.html
#
# Fixed: self.fc1 = nn.Linear(64 * image_size * image_size, 128)
#
class FashionCNN(nn.Module):
    def __init__(self, image_size = 28, in_channels = 1, out_channels = 10):
        super(FashionCNN, self).__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * image_size * image_size, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # 2x2最大池化
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        # 2x2最大池化
        x = F.max_pool2d(x, 2)
        # print("x.shape = ", x.shape)
        # 将卷积后的特征图展平，以便输入全连接层
        x = x.view(-1, 64 * self.image_size * self.image_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # 使用 log_softmax 激活函数进行分类概率计算
        return F.log_softmax(x, dim=1)

# 检查是否有可用的GPU，并定义设备（CPU或GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 将模型移动到设备上（CPU或GPU）
# 如果 CNN 使用了 max_pool2d 最大池化，那么 image_size 由实际输出的大小为准，
# 不使用最大池化的话，则使用原始的图片的 image_size 大小。
net = FashionCNN(7).to(device)

# 定义损失函数为：交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器为随机梯度下降（SGD）优化器，设置学习率为0.001，动量为0.9
# optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

def extend_label(label, output_dim):
    extended_label = np.zeros(output_dim, dtype=float)
    extended_label[label.item()] = 1.0
    return extended_label

def extend_labels(labels, output_dim):
    extended_labels = []
    for label in labels:
        extended_labels.append(extend_label(label, output_dim))
    return torch.tensor(np.array(extended_labels, dtype=float), dtype=torch.float)

# 4. 训练模型
print('batch_size = %d\n' % batch_size)
per_step = 256 / batch_size * 16
for epoch in range(num_epochs):
    # 设置模型为训练模式
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_iter):
        # 将数据和标签移动到设备上（CPU或GPU）
        inputs, labels = data[0].to(device), data[1].to(device)

        '''
        if cnt < 2:
            print("i = %d" % (i + 1))
            print('inputs.size(0) = %d' % inputs.size(0))
            print('labels.size(0) = %d' % labels.size(0))
            print('inputs.shape = ', inputs.shape)
            print('labels.shape = ', labels.shape)
            print(inputs)
            print(labels)
        '''

        # 将梯度清零
        optimizer.zero_grad()

        # 前向传播，获取预测输出
        outputs = net(inputs)

        # 计算损失值
        loss = criterion(outputs, labels)

        # 反向传播，计算梯度值
        loss.backward()

        # 更新权重参数
        optimizer.step()

        # 累加损失值
        running_loss += loss.item()
        if (step % per_step) == (per_step - 1):
            print("step = %d (%d)" % ((step + 1), (step + 1) * batch_size))
    # 输出当前轮次的平均损失值
    print('Epoch %d/%d loss: %.5f' % (epoch + 1, num_epochs, running_loss / len(train_iter)))

# 5. 验证测试集

# 设置模型为评估模式，关闭dropout和batch normalization等在训练模式下的特殊操作
net.eval()

correct = 0
total = 0

# 不需要计算梯度，以提高评估速度
with torch.no_grad():
    for i, data in enumerate(test_iter):
        # 将数据和标签移动到设备上（CPU或GPU）
        images, labels = data[0].to(device), data[1].to(device)
        # 前向传播，获取预测输出
        outputs = net(images)
        # 获取最大概率对应的类别标签作为预测结果
        _, predicted = torch.max(outputs.data, 1)
        # 统计样本总数
        total += labels.size(0)
        # 统计正确分类的样本数量
        correct += (predicted == labels).sum().item()

# 输出模型在测试数据集上的准确率
print('Accuracy of the network on the test images: %0.2f %%' % (
        100.0 * correct / total))

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for x, y in data_iter:
        acc_sum += (net(x).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def train_ch3(net, train_iter, test_iter, loss_func, num_epochs, optimizer):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for x, y in train_iter:
            y_hat = net(x)
            loss = loss_func(y_hat, y).sum()

            # 梯度清零
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_l_sum += loss.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def use_svg_display():
    """Use svg format to display plot in jupyter"""
    # DeprecationWarning: `set_matplotlib_formats` is deprecated since IPython 7.23
    # display.set_matplotlib_formats('svg')
    import matplotlib_inline
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

def show_fashion_mnist(images, labels):
    use_svg_display()

    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    # 如果在 jupyter notebook 里则注释这一句
    plt.show()

def show_figure(net, test_iter):
    for x, y in test_iter:
        break

    true_labels = get_fashion_mnist_labels(y)
    pred_labels = get_fashion_mnist_labels(net(x).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

    show_fashion_mnist(x[0:9], titles[0:9])

show_figure(net, test_iter)

'''
Output:

batch_size = 256

Epoch 1 loss: 0.597
Epoch 2 loss: 0.359
Epoch 3 loss: 0.310
Epoch 4 loss: 0.278
Epoch 5 loss: 0.256
Epoch 6 loss: 0.239
Epoch 7 loss: 0.222
Epoch 8 loss: 0.211
Epoch 9 loss: 0.200
Epoch 10 loss: 0.189

Accuracy of the network on the test images: 91 %

----------------------------------
Output:

batch_size = 64

Epoch 1/10 loss: 0.45434
Epoch 2/10 loss: 0.29076
Epoch 3/10 loss: 0.24785
Epoch 4/10 loss: 0.21675
Epoch 5/10 loss: 0.19337
Epoch 6/10 loss: 0.17104
Epoch 7/10 loss: 0.15336
Epoch 8/10 loss: 0.13416
Epoch 9/10 loss: 0.11915
Epoch 10/10 loss: 0.10145

Accuracy of the network on the test images: 92.10 %

'''
