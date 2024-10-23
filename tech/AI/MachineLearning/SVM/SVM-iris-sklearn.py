#
# From: https://blog.csdn.net/Like_July_moon/article/details/136750962
#
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# 加载数据集
iris = datasets.load_iris()
x = iris.data
y = iris.target

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# 特征缩放
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#
# 创建 SVM 分类器实例, 使用线性核
#
# 选项 kernel:
#   'linear':       线性核函数
#   'poly':         多项式核函数
#   'rbf':          径向基函数/高斯核, 默认值
#   'sigmod':       sigmod核函数
#   'precomputed':  核矩阵, 表示自己提前计算好核函数矩阵
#
# 选项 C: float类型, 默认值为1.0, 错误项的惩罚系数
#
# 选项 tol: float类型, 默认值为: 1e^-3
#
svm_classifier = SVC(kernel='rbf')

# 训练模型
svm_classifier.fit(x_train, y_train)

# 预测
y_predict = svm_classifier.predict(x_test)

# 评估模型
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))

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

'''