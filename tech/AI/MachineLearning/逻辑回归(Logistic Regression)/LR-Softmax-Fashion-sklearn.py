#
# From: https://blog.csdn.net/weixin_38169413/article/details/103598534
#
import numpy as np
import torch
import torchvision
from torch import nn
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
# from d2l import torch as d2l
from matplotlib import pyplot as plt
import sys

# -----------------------------获取并读取FashionMNIST数据集函数，返回小批量train，test-----------------------------------
def load_data_fashion_mnist(batch_size, root='~/.datasets/FashionMNIST'):
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

num_epochs = 20
lr = 0.1

def load_data_and_label(dataset_iter, input_dim, output_dim):
    data_arr = []
    label_arr = []
    data_set_len = 0
    for batch_data, batch_label in dataset_iter:
        for data in batch_data:
            flat_data = data.view(input_dim)
            data_arr.append(flat_data.numpy())
        for label in batch_label:
            # extend_label = np.zeros(output_dim, dtype=float)
            # extend_label[label.item()] = 1.0
            label_arr.append(label.item())
            # label_arr.append(np.array(label.view(1).numpy()))
            # label_arr.append(np.array(extend_label))
        data_set_len = data_set_len + 1
    print("data_set_len = ", data_set_len)
    return np.array(data_arr), np.array(label_arr)

train_data, train_label = load_data_and_label(train_iter, num_inputs, num_outputs)
test_data, test_label = load_data_and_label(test_iter, num_inputs, num_outputs)

print(train_data.shape)
print(train_label.shape)

std = StandardScaler()
train_data = std.fit_transform(train_data)
test_data  = std.fit_transform(test_data)

# solver='lbfgs', solver='liblinear', multi_class="multinomial"， max_iter=num_epochs
# multi_class‌：多分类策略，'ovr' 表示一对一，'multinomial' 表示多类逻辑回归。
# 其中 liblinear 不支持 'multinomial' 模式。
model_softmax_regression = linear_model.LogisticRegression(solver='lbfgs', multi_class="multinomial", penalty='l2', C=1, max_iter=num_epochs)
model_softmax_regression.fit(train_data, train_label)

y_predict = model_softmax_regression.predict(test_data)
accurcy = np.sum(y_predict == test_label) / len(test_data)

print("")
print("accurcy = %0.4f" % accurcy)
print("")

##
## From: https://blog.csdn.net/m0_47256162/article/details/135439913
##

# 8. 单独打印训练集和测试集的精度
train_accuracy = accuracy_score(train_label, model_softmax_regression.predict(train_data))
test_accuracy  = accuracy_score(test_label, model_softmax_regression.predict(test_data))
print(f"Train集精度: {train_accuracy: .4f}")
print(f"Test 集精度: {test_accuracy: .4f}")
print("")

# 9. 打印训练集和测试集的模型评估指标
train_predictions = model_softmax_regression.predict(train_data)
test_predictions  = model_softmax_regression.predict(test_data)

print("网格搜索后的训练集评估指标：")
print(classification_report(train_label, train_predictions))

print("网格搜索后的测试集评估指标：")
print(classification_report(test_label, test_predictions))
