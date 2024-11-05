import torch
from matplotlib import pyplot as plt
# %matplotlib inline

x = torch.arange(-6.0, 6.0, 0.1, requires_grad=True)
sigmoid = torch.nn.Sigmoid()
y = sigmoid(x)

# 解决中文乱码问题
# plt.rcParams['font.family'] = 'SimHei'
plt.figure(figsize=(16, 2.1))
plt.title('sigmoid(x)')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.ylim(-0.3, 1.3)
plt.xticks([-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6])
plt.yticks([0,0.25,0.5,0.75,1.0])
plt.grid()
plt.plot(x.detach(), y.detach(), color='blue', label='sigmoid(x)', linewidth=2)
plt.legend(['sigmoid(x)'], loc='upper left')
plt.show()
