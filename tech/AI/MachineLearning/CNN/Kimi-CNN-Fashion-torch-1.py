#
# From Kimi chat
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义CNN模型
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输入通道1，输出通道32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 输入通道32，输出通道64
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 展平后连接到128个节点的全连接层
        self.fc2 = nn.Linear(128, 10)  # 连接到10个节点的全连接层（10个类别）

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 第一个卷积层后接ReLU激活函数
        x = F.max_pool2d(x, 2)  # 2x2最大池化
        x = F.relu(self.conv2(x))  # 第二个卷积层后接ReLU激活函数
        x = F.max_pool2d(x, 2)  # 2x2最大池化
        x = x.view(-1, 64 * 7 * 7)  # 展平
        x = F.relu(self.fc1(x))  # 第一个全连接层后接ReLU激活函数
        x = self.fc2(x)  # 第二个全连接层
        return F.log_softmax(x, dim=1)  # 应用LogSoftmax

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64

# 加载数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='~/.datasets/FashionMNIST', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='~/.datasets/FashionMNIST', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 设置最大训练轮数
num_epochs = 5
# 学习率
lr = 0.001

# 初始化模型并移动到设备
model = FashionCNN().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 训练模型
print('batch_size = %d\n' % batch_size)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # 梯度清零
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100.0 * correct / total}%')

'''
Output:

batch_size = 64

Epoch 1/5, Loss: 0.4340517887714575
Epoch 2/5, Loss: 0.27753355493074033
Epoch 3/5, Loss: 0.22975117758091199
Epoch 4/5, Loss: 0.1998116783797741
Epoch 5/5, Loss: 0.1726138395374454
Accuracy of the network on the test images: 91.74%

'''
