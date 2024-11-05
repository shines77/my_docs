import torch
from matplotlib import pyplot as plt
# %matplotlib inline

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
leaky_relu = torch.nn.LeakyReLU(negative_slope=0.1)
y = leaky_relu(x)

# 解决中文乱码问题
# plt.rcParams['font.family'] = 'SimHei'
plt.figure(figsize=(8, 4))
plt.title('Leaky-ReLU(x)')
plt.xlabel('x')
plt.ylabel('Leaky-ReLU(x)')
plt.grid()
plt.plot(x.detach(), y.detach(), color='blue', label='Leaky-ReLU(x)', linewidth=2)
plt.legend(['Leaky-ReLU(x)'], loc='upper left')
plt.show()
