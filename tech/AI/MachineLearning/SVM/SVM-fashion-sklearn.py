#
# From: https://blog.csdn.net/Like_July_moon/article/details/136750962
# From: https://blog.csdn.net/weixin_41362649/article/details/129850761
#
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from matplotlib import pyplot as plt
import sys

def split_fashion_data(data_loader, image_size, batch_size, limit_size):
    image_list = []
    label_list = []
    data_size = 0
    for images, labels in data_loader:
        for image in images:
            image = image.view(image_size * image_size)
            image_list.append(image.numpy())
        for label in labels:
            label_list.append(label.numpy())
        data_size += batch_size
        if (data_size >= limit_size):
            break

    return np.array(image_list), np.array(label_list)

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
    上面的 mnist_train, mnist_test 都是 torch.utils.data.Dataset 的子类，所以可以使用len()获取数据集的大小
    训练集和测试集中的每个类别的图像数分别是 60000，1000，两个数据集分别有10个类别
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

    x_train, y_train = split_fashion_data(train_iter, image_size, batch_size, 60000 / 5)
    print('x_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('')
    x_test, y_test = split_fashion_data(test_iter, image_size, batch_size, 10000 / 5)
    print('x_test.shape = ', x_test.shape)
    print('y_test.shape = ', y_test.shape)
    print('')

    return x_train, y_train, x_test, y_test

image_size = 28
batch_size = 64

print('')
print('batch_size = %d\n' % batch_size)

# 载入 fashion 图像数据和标签, train 和 test
x_train, y_train, x_test, y_test = load_data_fashion_mnist(batch_size, image_size)

# 特征缩放
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#
# 创建 SVM 分类器实例, 使用线性核 或 径向基
#
# See: https://www.cnblogs.com/cgmcoding/p/13559984.html
#
# 选项 kernel:
#   'linear':       线性核函数
#   'poly':         多项式核函数
#   'rbf':          径向基函数/高斯核, 默认值
#   'sigmod':       sigmod核函数
#   'precomputed':  核矩阵, 表示自己提前计算好核函数矩阵
#
# 选项 ganma: 可选 'scale'、'auto' 或 float类型, 默认值为 `scale`, gamma 取值为 1 / (n_features * X.var())
# 选项 C: float类型, 默认值为1.0, 正则化参数，正则化的强度与C成反比
#
# 选项 tol: float类型, 算法停止的条件，默认为0.001
#
svm_classifier = SVC(kernel='rbf')

# 训练模型
svm_classifier.fit(x_train, y_train)

# 预测
y_predict = svm_classifier.predict(x_test)

# 评估模型
print(confusion_matrix(y_test, y_predict))
print('')
print(classification_report(y_test, y_predict))

# 输出训练集的准确率
print('(train) evaluate_accuracy = %0.2f %%' % (svm_classifier.score(x_train, y_train) * 100.0))
print('')
print('(test) evaluate_accuracy = %0.2f %%' % (svm_classifier.score(x_test, y_test) * 100.0))
print('')

#
# 验证测试集
#
# From: https://cloud.tencent.cn/developer/article/2129747
#
y_train_hat = svm_classifier.predict(x_train)
y_train_1d = y_train.reshape((-1))
comp = zip(y_train_1d[:20], y_train_hat[:20])
print(list(comp))
print('')

y_test_hat = svm_classifier.predict(x_test)
y_test_1d = y_test.reshape((-1))
comp = zip(y_test_1d[:20], y_test_hat[:20])
print(list(comp))
print('')

# 绘制可视化图像
plt.figure()
plt.subplot(121)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train.reshape((-1)), edgecolors='k',s=50)
plt.subplot(122)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train_hat.reshape((-1)), edgecolors='k',s=50)
plt.show()

'''
Output:

batch_size = 64

# 原始数据集长度, 训练时间有点长, 半个小时以上
x_train.shape =  (60000, 784)
y_train.shape =  (60000,)

x_test.shape =  (10000, 784)
y_test.shape =  (10000,)

[[856   0   9  29   3   1  90   0  12   0]
 [  4 961   3  25   3   0   4   0   0   0]
 [ 13   1 815  12  92   0  64   0   3   0]
 [ 28   3  12 893  31   0  29   0   4   0]
 [  0   0  84  30 821   0  61   0   4   0]
 [  0   0   0   1   0 958   0  26   2  13]
 [142   1  92  31  67   0 647   0  20   0]
 [  0   0   0   0   0  20   0 959   0  21]
 [  3   0   2   5   4   4   3   4 974   1]
 [  0   0   0   0   0   8   0  37   3 952]]

              precision    recall  f1-score   support

           0       0.82      0.86      0.84      1000
           1       0.99      0.96      0.98      1000
           2       0.80      0.81      0.81      1000
           3       0.87      0.89      0.88      1000
           4       0.80      0.82      0.81      1000
           5       0.97      0.96      0.96      1000
           6       0.72      0.65      0.68      1000
           7       0.93      0.96      0.95      1000
           8       0.95      0.97      0.96      1000
           9       0.96      0.95      0.96      1000

    accuracy                           0.88     10000
   macro avg       0.88      0.88      0.88     10000
weighted avg       0.88      0.88      0.88     10000

evaluate_accuracy = 92.37 %

-------------------------------------------------

batch_size = 64

#
# 训练集和测试集减少到原本的 1/5, 训练时间就快了很多
#
x_train.shape =  (12032, 784)
y_train.shape =  (12032,)

x_test.shape =  (2048, 784)
y_test.shape =  (2048,)

[[167   0   1  10   1   0  22   0   2   0]
 [  0 198   0  11   0   0   0   0   0   0]
 [  2   0 173   3  25   0  16   0   1   0]
 [  4   1   0 178   4   0   9   0   1   0]
 [  0   0  23   5 171   0  20   0   4   0]
 [  0   0   0   0   0 184   0  11   0   4]
 [ 28   0  21   4  14   1 127   0   6   0]
 [  0   0   0   0   0   5   0 196   0   4]
 [  0   0   1   1   0   1   3   0 193   0]
 [  0   0   0   0   0   3   0   8   2 179]]

              precision    recall  f1-score   support

           0       0.83      0.82      0.83       203
           1       0.99      0.95      0.97       209
           2       0.79      0.79      0.79       220
           3       0.84      0.90      0.87       197
           4       0.80      0.77      0.78       223
           5       0.95      0.92      0.94       199
           6       0.64      0.63      0.64       201
           7       0.91      0.96      0.93       205
           8       0.92      0.97      0.95       199
           9       0.96      0.93      0.94       192

    accuracy                           0.86      2048
   macro avg       0.86      0.86      0.86      2048
weighted avg       0.86      0.86      0.86      2048

evaluate_accuracy = 92.10 %

'''