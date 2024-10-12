#
# From: https://github.com/TrWestdoor/pytorch-practice/blob/master/rnn_classify.py
#

import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms

INPUT_SIZE = 28
BATCH_SIZE = 1
EPOCH = 1
LR = 0.005
DOWNLOAD_MNIST = False

train_data = datasets.MNIST(
    root='./MNIST',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_data = datasets.MNIST(root='./MNIST', train=False, transform=transforms.ToTensor())
test_x = test_data.data.type(torch.FloatTensor)[:2000] / 255
test_y = test_data.targets.numpy()[:2000]

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        self.out = nn.Linear(self.rnn.hidden_size, 10)

    def forward(self, x):
        r_out, (h_c, h_h) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out

rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_fun = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.view(-1, 28, 28)

        r_out = rnn(b_x)
        loss = loss_fun(r_out, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step % 50) == 0:
            test_out = rnn(test_x)
            pred_y = torch.max(test_out, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print("Epoch: ", epoch, "| train loss: ", loss.data.numpy(), "| test accuracy: %.2f %%" % (accuracy * 100.0))
