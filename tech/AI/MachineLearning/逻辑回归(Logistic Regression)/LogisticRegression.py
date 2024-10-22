#
# From: https://blog.csdn.net/Like_July_moon/article/details/136750962
#

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 加载数据集
iris = load_iris()
x = iris.data
y = iris.target

# 为了演示二分类，我们只选择两个类
x = x[y != 2]
y = y[y != 2]

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# 特征缩放
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 创建逻辑回归分类器实例
log_reg = LogisticRegression()

# 训练模型
log_reg.fit(x_train, y_train)

# 预测
y_pred = log_reg.predict(x_test)

# 评估模型
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
