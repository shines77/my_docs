#
# From: https://zhuanlan.zhihu.com/p/31054600
#
import math
import matplotlib.pyplot as plt
import numpy as np

def sigmoid_(x):
    return 1.0 / (1.0 + np.exp(-x))

def d_sigmoid_(x):
    return sigmoid_(x) * (1 - sigmoid_(x))

def newfun_(x):
    return 2 * sigmoid_(x) - 1

def d_newfun_(x):
    return (1 - newfun_(x) * newfun_(x)) / 2

def tanh_(x):
    return 2 * sigmoid_(2 * x) - 1

def d_tanh_(x):
    return 1 - tanh_(x) * tanh_(x)

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)

x = np.linspace(-5, 5)
sigmoid = sigmoid_(x)
d_sigmoid = d_sigmoid_(x)
newfun = newfun_(x)
d_newfun = d_newfun_(x)
tanh = tanh_(x)
d_tanh = d_tanh_(x)

plt.xlim(-5,5)
plt.ylim(-1.1, 1.1)

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.set_xticks([-5, -1, 0, 1, 5])
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
ax.set_yticks([-1, 1])

#plt.plot(x, sigmoid, label="Sigmoid", color="blue")
#plt.plot(x, d_sigmoid, label="Sigmoid'", color="green")
#plt.plot(x, newfun, label="Newfun", color="blue")
#plt.plot(x, d_newfun, label="Newfun'", color="green")
plt.plot(x, tanh, label="Tanh", color="blue")
plt.plot(x, d_tanh, label="Tanh'", color="green")
plt.legend()
plt.show()
