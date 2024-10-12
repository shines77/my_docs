#
#
# pytorch学习笔记(2)：在 MNIST 上实现一个 cnn
#
# From: https://mp.weixin.qq.com/s?__biz=MzAwMDgyNjE3OA==&mid=2247483868&idx=1&sn=ee395d612522e95a48592170c05bf814&chksm=9ae24ff1ad95c6e7cc5c0d561ebdb8c8826e6b0f0c355ad42e49028d6677dfea8d97bb876cf6&scene=21#wechat_redirect
#
# 10 分钟完全读懂 PyTorch
# From: https://mp.weixin.qq.com/s?__biz=MzAxMjUyNDQ5OA==&mid=2653563579&idx=2&sn=f2a3c115977af6368b37cb03676b3771&chksm=806e0406b7198d1065236238eb8d34bdd62011ad9d5fd845879395071860e9974ef61da151a5&scene=27
#

import numpy as np
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision
import time

BATCH_SIZE = 32
EPOCH = 50

class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = torch.nn.RNN(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        self.out = torch.nn.Linear(self.rnn.hidden_size, self.rnn.input_size)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state

def main():
    # 0. Get started
    print("\nBegin simple MNIST CNN Dataset with PyTorch demo.\n")
    torch.manual_seed(1)
    np.random.seed(1)

    seq_size = 100
    input_size = 1
    hidden_dim = 32
    layer_dim = 1

    steps = np.linspace(0, np.pi * 2, seq_size, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    # print("size(x_np) = ", np.size(x_np))
    # print(x_np)

    x_np = torch.from_numpy(np.sin(steps))
    y_np = torch.from_numpy(np.cos(steps))

    train_x = Variable(torch.zeros(seq_size, hidden_dim, input_size))
    train_y = Variable(torch.zeros(seq_size, hidden_dim, input_size))

    for seq in range(seq_size):
        for h in range(hidden_dim):
            for i in range(input_size):
                train_x[seq][h][i] = x_np[seq]
                train_y[seq][h][i] = y_np[seq]

    # print(train_x)

    # Initialize hidden state with zeros
    h_state = Variable(torch.zeros(layer_dim, seq_size, hidden_dim, dtype=torch.float32))
    # h_state = torch.from_numpy(np.zeros((layer_dim, seq_size, hidden_dim), dtype=np.float32))
    # print("h_state.dim() = ", h_state.dim())
    # print(h_state)

    learning_rate = 0.05
    rnn = RNN()

    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()

    print("Starting training...")

    time_start = time.time()
    output, h_n = rnn(train_x, h_state)
    loss = loss_func(output, train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    h_state = h_n
    print('Epoch: ', 0, '| train loss: %.4f' % loss.data, '| test accuracy: %.2f %%' % 0.0)

    print("Training complete \n")
    return

    for epoch in range(EPOCH):
        output, h_n = rnn(train_x, h_state)
        loss = loss_func(output, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        h_state = h_n

        # 打印训练过程
        if epoch % 10 == 0:
            accuracy = 0.0
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data, '| test accuracy: %.2f %%' % (accuracy * 100.0))

    print("Training complete \n")

if __name__ == "__main__":
    main()
