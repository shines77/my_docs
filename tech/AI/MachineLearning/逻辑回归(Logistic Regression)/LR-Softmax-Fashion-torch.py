#
# From: https://blog.csdn.net/m0_53881899/article/details/140667795
#
import torch
import torchvision
from torch import nn
# from d2l import torch as d2l
from matplotlib import pyplot as plt
import sys

# -----------------------------获取并读取FashionMNIST数据集函数，返回小批量train，test-----------------------------------
def load_data_fashion_mnist(batch_size, root='./datasets/FashionMNIST'):
    trans = []
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root,
                                                    train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root,
                                                   train=False, download=True, transform=transform)
    '''
    上面的 mnist_train,mnist_test 都是 torch.utils.data.Dataset 的子类，所以可以使用len()获取数据集的大小
    训练集和测试集中的每个类别的图像数分别是6000，1000，两个数据集分别有10个类别
    '''
    # mnist 是 torch.utils.data.dataset 的子类，因此可以将其传入torch.utils.data.DataLoader来创建一个DataLoader实例来读取数据
    # 在实践中，数据读取一般是训练的性能瓶颈，特别是模型较简单或者计算硬件性能比较高的时候
    # DataLoader一个很有用的功能就是允许多进程来加速读取  使用num_works来设置4个进程读取数据
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter

# 1. 获取训练集和测试集的DataLoader迭代器
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# 2. 定义网络模型
num_inputs = 784    # 由于softmax回归输入要求是向量，所以对于28×28的图片，应该拉长成784的向量作为输入
num_outputs = 10    # 10分类问题，输出为长度为10的向量，里面记录样本在各个类别上的预测概率

net = nn.Sequential(
    # 将输入展平
    nn.Flatten(),
    # 线性全连接层
    nn.Linear(num_inputs, num_outputs)
)

# 3. 初始化参数。这个函数检查 m 是否是一个线性层（nn.Linear），
# 如果是，则使用正态分布（均值为0，标准差为0.01）初始化该层的权重参数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

# 4. 交叉熵损失函数
loss_func = nn.CrossEntropyLoss()

# 5. 梯度下降算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

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

# 6. 训练过程
num_epochs = 10
train_ch3(net, train_iter, test_iter, loss_func, num_epochs, trainer)

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
