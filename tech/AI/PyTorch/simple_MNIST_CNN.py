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
import torchvision
import time

BATCH_SIZE = 32
EPOCH = 5

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.classify = torch.nn.Sequential(
            torch.nn.Linear(n_feature, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_output),
        )

    def forward(self, x_layer):
        # x_layer = torch.relu(self.n_hidden(x_layer))
        # x_layer = self.out_layer)
        x_layer = self.classify(x_layer)
        x_layer = torch.nn.functional.softmax(x_layer)
        return x_layer

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 5, 1, 2),
            torch.nn.ReLU()
        )

        self.out = torch.nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

def main():
    # 0. Get started
    print("\nBegin simple MNIST CNN Dataset with PyTorch demo.\n")
    torch.manual_seed(1)
    np.random.seed(1)

    train_data = torchvision.datasets.MNIST(
        root='./mnist',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    test_data = torchvision.datasets.MNIST(
        root='./mnist',
        train=False,
        transform=torchvision.transforms.ToTensor()
    )
    test_x = torch.unsqueeze(test_data.data, dim=1) / 255.
    test_y = test_data.targets

    LR = 0.02
    cnn = CNN()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = torch.nn.CrossEntropyLoss()

    print("Starting training...")

    time_start = time.time()
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):
            output = cnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 打印训练过程
            if step % 50 == 0:
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()

                accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data, '| test accuracy: %.2f%%' % (accuracy * 100.0))

    print("Training complete \n")

if __name__ == "__main__":
    main()
