import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
# %matplotlib inline

def tanh_simple(x):
    xx = copy.deepcopy(x)
    xx = xx.detach().numpy()
    exp_neg_2x = np.exp(-2 * xx)
    result = (1 - exp_neg_2x) / (1 + exp_neg_2x)
    return torch.tensor(result)

def tanh_fast(x):
    exp_2x = np.exp(2 * x)
    return (1 - 2 / (1 + exp_2x))

def tanh_fast2(x):
    exp_neg_2x = np.exp(-2 * x)
    return (2 / (1 + exp_neg_2x) - 1)

x1 = torch.arange(-4.0, 4.0, 0.1, requires_grad=True)
x2 = np.linspace(-4, 4, 800)
tanh = torch.nn.Tanh()
y1 = tanh(x1)
y2 = tanh_fast(x2)

# 解决中文乱码问题
# plt.rcParams['font.family'] = 'SimHei'
plt.figure(figsize=(14, 4.2))
plt.title('tanh(x)')
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.ylim(-1.3, 1.3)
plt.xticks([-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6])
plt.yticks([-1.0,-0.5,0,0.5,1.0])
plt.grid()
plt.plot(x1.detach(), y1.detach(), color='blue', label='tanh(x)', linewidth=2)
# plt.plot(x2, y2, color='blue', label='tanh(x)', linewidth=2)
plt.legend(['tanh(x)'], loc='upper left')
plt.show()
