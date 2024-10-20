#
# From: https://www.jianshu.com/p/31e802802735
#

from sklearn import linear_model
import pandas as pd
import numpy as np
# `load_boston` has been removed from scikit-learn since version 1.2.
# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml

# `load_boston` has been removed from scikit-learn since version 1.2.
'''
boston = load_boston()
datas = boston.data
target = boston.target
name_data = boston.feature_names
'''

#
# See: https://blog.csdn.net/2301_76161684/article/details/139895701
#
boston = fetch_openml(name="boston", version=1, as_frame=False)
datas = boston['data']
target = boston['target']
name_data = boston['feature_names']

'''
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r'\\s{2,}', skiprows=7, nrows=14, header=None, engine='python')
name_data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
print(name_data)

raw_df = pd.read_csv(data_url, sep=r'\\s+', skiprows=22, header=None, engine='python')
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
'''

fig = plt.figure()
fig.set_size_inches(14, 9)
for i in range(13):
    ax = fig.add_subplot(4, 4, i+1)
    x = datas[:, i]
    y = target
    plt.title(name_data[i])
    ax.scatter(x, y)
plt.tight_layout()  # 自动调整子图间距
plt.show()

j_ = []
for i in range(13):
    if name_data[i] == 'RM':
        continue
    if name_data[i] =='LSTAT':
        continue
    j_.append(i)
x_data = np.delete(datas, j_, axis=1)

X_train, X_test, y_train, y_test = train_test_split(x_data, target, random_state=0, test_size=0.20)

lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)
lr_y_predict = lr.predict(X_test)
score = lr.score(X_test, y_test)

print(score)
print("w0: ", lr.intercept_)
print("w1, w2: ", lr.coef_)
