#
# 10 分钟完全读懂 PyTorch
# From: https://mp.weixin.qq.com/s?__biz=MzAxMjUyNDQ5OA==&mid=2653563579&idx=2&sn=f2a3c115977af6368b37cb03676b3771&chksm=806e0406b7198d1065236238eb8d34bdd62011ad9d5fd845879395071860e9974ef61da151a5&scene=27
#
# 实现一个简单的分类器
#

import numpy as np
import torch

class Batch:
    def __init__(self, num_items, batch_size, seed=0):
        self.num_items = num_items; self.batch_size = batch_size
        self.rnd = np.random.RandomState(seed)
    def next_batch(self):
        return self.rnd.choice(self.num_items, self.batch_size,
            replace=False)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.n_hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x_layer):
        x_layer = torch.relu(self.n_hidden(x_layer))
        x_layer = self.out(x_layer)
        x_layer = torch.nn.functional.softmax(x_layer)
        return x_layer

def accuracy(model, optimizer, loss_func, data_x, data_y):
    X = torch.Tensor(data_x)
    Y = torch.Tensor(data_y)
    optimizer.zero_grad()
    oupt = model(X)
    # print("oupt = {}", oupt)

    loss = loss_func(oupt, Y)
    loss.backward()

    # (_, arg_maxs) = torch.max(oupt.data, dim=1)
    # print("arg_maxs = {}", arg_maxs)
    # num_correct = torch.sum(Y==arg_maxs)
    # acc = (num_correct * 100.0 / len(data_y))
    # return acc.item()
    return loss.item()

def main():
    # 0. Get started
    print("\nBegin simple classifier Dataset with PyTorch demo.\n")
    torch.manual_seed(1)
    np.random.seed(1)

    # 1. Generating data
    print("Generating data into matrix.\n")
    _x1 = torch.rand(size=(100, 2)) * 4.0
    _x2 = torch.rand(size=(100, 2)) * -4.0
    train_x = torch.cat([_x1, _x2], 0)
    # print(train_x)
    _y1 = torch.ones([100, 1])
    _y2 = torch.zeros([100, 1])
    train_y1 = torch.cat([_y1, _y2], dim=1)
    _y1 = torch.zeros([100, 1])
    _y2 = torch.ones([100, 1])
    train_y2 = torch.cat([_y1, _y2], dim=1)
    train_y = torch.cat([train_y1, train_y2], dim=0)
    # print(train_y)

    _x1 = torch.rand(size=(100, 2)) * -4.0
    _x2 = torch.rand(size=(100, 2)) * 4.0
    test_x = torch.cat([_x1, _x2], 0)
    _y1 = torch.zeros([100, 1])
    _y2 = torch.ones([100, 1])
    test_y1 = torch.cat([_y1, _y2], dim=1)
    _y1 = torch.ones([100, 1])
    _y2 = torch.zeros([100, 1])
    test_y2 = torch.cat([_y1, _y2], dim=1)
    test_y = torch.cat([train_y1, train_y2], dim=0)

    net = Net(n_feature=2, n_hidden=10, n_output=2)
    # print(net)

    # set training mode
    net = net.train()

    lrn_rate = 0.01; batch_size = 12
    max_i = 1600; n_items = len(train_x)

    optimizer = torch.optim.SGD(net.parameters(), lr=lrn_rate)
    batcher = Batch(num_items=n_items, batch_size=batch_size)
    loss_func = torch.nn.CrossEntropyLoss()

    print("Starting training...")

    for i in range(1, max_i):
        if (i > 0) and (i % 100 == 0):
            print("iteration = %4d" % i, end="")
            print("  loss = %7.4f" % loss.item(), end="\n")
        curr_bat = batcher.next_batch()
        X = torch.Tensor(train_x[curr_bat])
        Y = torch.Tensor(train_y[curr_bat])
        optimizer.zero_grad()
        oupt = net(X)
        loss = loss_func(oupt, Y)
        loss.backward()
        optimizer.step()

    print("Training complete \n")

    # 4. evaluate model
    # set eval mode
    net = net.eval()
    acc = accuracy(net, optimizer, loss_func, test_x, test_y)
    # print("Accuracy on test data = %0.2f%%\n" % acc)
    print("Loss on test data = %7.4f\n" % acc)

    acc = accuracy(net, optimizer, loss_func, train_x, train_y)
    print("Loss on train data = %7.4f\n" % acc)

if __name__ == "__main__":
    main()
