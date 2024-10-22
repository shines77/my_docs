#
# From： https://blog.csdn.net/weixin_45666566/article/details/107595200
#
import torch
import torchvision
import numpy as np
# from IPython import display
from matplotlib import pyplot as plt

import zipfile
import sys
#
# d2lzh_pytorch 包下载: https://pan.baidu.com/s/179Vx8CTQR4f-4hkXCORI3A?pwd=n88d
#
# 加上 d2lzh_pytorch 的路径
# sys.path.append('E:\d2lzh_pytorch')
# import d2lzh_pytorch as d2l

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
    上面的 mnist_train, mnist_test 都是 torch.utils.data.Dataset 的子类，所以可以使用len()获取数据集的大小
    训练集和测试集中的每个类别的图像数分别是 6000，1000，两个数据集分别有10个类别
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

# 小批量进行读取
batch_size = 256

#
# Fashion MNIST 数据集: https://github.com/zalandoresearch/fashion-mnist
#
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

# 随机初始化参数
W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

def softmax(x):
    x_exp = x.exp()
    # 行元素求和
    partition = x_exp.sum(dim=1, keepdim=True)
    # 这里应用了广播机制
    return x_exp / partition

def net(x):
    return softmax(torch.mm(x.view((-1, num_inputs)), W) + b)

def sgd(params, lr, batch_size):
    # 为了和原书保持一致，这里除以了batch_size，但是应该是不用除的，因为一般用PyTorch计算loss时就默认已经
    # 沿batch维求了平均了。
    for param in params:
        # 这里必须除以 batch_size
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data

def cross_entropy(y_hat, y):
    # About x.gather() function: https://blog.csdn.net/u013250861/article/details/139223852
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))

def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for x, y in data_iter:
        acc_sum += (net(x).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

num_epochs = 5
lr = 0.1

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for x, y in train_iter:
            y_hat = net(x)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                # "softmax回归的简洁实现" 一节将用到
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

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
