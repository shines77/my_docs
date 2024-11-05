import torch
from matplotlib import pyplot as plt
# %matplotlib inline

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
relu = torch.nn.ReLU()
y = relu(x)

# 解决中文乱码问题
# plt.rcParams['font.family'] = 'SimHei'
plt.figure(figsize=(8, 4))
plt.title('ReLU(x)')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.grid()
plt.plot(x.detach(), y.detach(), color='blue', label='ReLU(x)', linewidth=2)
plt.legend(['ReLU(x)'], loc='upper left')
plt.show()
