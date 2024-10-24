#
# From: https://blog.csdn.net/Like_July_moon/article/details/136750962
#
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt

# 加载数据集
iris = datasets.load_iris()
x = iris.data
y = iris.target

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

print('')
print('x_train.shape ', x_train.shape)
print('y_train.shape ', y_train.shape)
print('')
print('x_train = ', x_train[:9])
print('y_train = ', y_train[:9])
print('')

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
comp = zip(y_train_1d, y_train_hat)
print(list(comp))

print('')
y_test_hat = svm_classifier.predict(x_test)
y_test_1d = y_test.reshape((-1))
comp = zip(y_test_1d, y_test_hat)
print(list(comp))

# 绘制可视化图像
plt.figure()
plt.subplot(121)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train.reshape((-1)), edgecolors='k',s=50)
plt.subplot(122)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train_hat.reshape((-1)), edgecolors='k',s=50)
plt.show()

'''
Output:

[[19  0  0]
 [ 0 13  0]
 [ 0  0 13]]
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        19
           1       1.00      1.00      1.00        13
           2       1.00      1.00      1.00        13

    accuracy                           1.00        45
   macro avg       1.00      1.00      1.00        45
weighted avg       1.00      1.00      1.00        45

evaluate_accuracy = 96.19 %

'''