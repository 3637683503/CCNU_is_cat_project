import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

# 检查是否有可用的GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda")
else:
    device = torch.device("cpu")
    print("cpu")

# 定义卷积神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Conv2d(1, 32, kernel_size=3)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=3)
        self.layer3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128 * 1 * 1, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)
        x = self.layer2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)
        x = self.layer3(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)
        batch_size = x.size(0)
        channels = 128
        height = x.size(2)
        width = x.size(3)
        x = x.view(batch_size, channels * height * width)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = nn.functional.log_softmax(x, 1)
        return x

# 计算模型判断正确率的函数
def evaluate(test_data, net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            x = x.to(device)
            y = y.to(device)
            # 去掉 x = x.view(-1, 28 * 28)
            outputs = net(x)
            _, predicted = torch.max(outputs, 1)
            n_correct += (predicted == y).sum().item()
            n_total += y.size(0)
    return n_correct / n_total

def train():
    '''
    训练模型，并打印出每次的正确率
    '''
    transform = transforms.Compose([transforms.ToTensor()])
    # 加载训练集
    train_data = DataLoader(
        dataset=MNIST('', train=True, transform=transform, download=True),
        batch_size=15,
        shuffle=True,
        drop_last=True
    )
    # 加载测试集
    test_data = DataLoader(
        dataset=MNIST('', train=False, transform=transform, download=True),
        batch_size=15,
        shuffle=False,
        drop_last=True
    )
    net = Net()
    net = net.to(device)  # 将模型移动到 GPU

    print('initial accuracy:', evaluate(test_data, net))

    rate = 0.001  # 初始化学习率
    for epoch in range(5):  # 假设训练 10 个 epoch
        optimizer = torch.optim.Adam(net.parameters(), lr=rate, weight_decay=1e-3)  # 添加 weight_decay 实现 L2 正则化
        for images, labels in train_data:
            images = images.to(device)  # 将数据移动到 GPU
            labels = labels.to(device)  # 将数据移动到 GPU
            optimizer.zero_grad()
            outputs = net(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)  # 计算损失
            loss.backward()  # 自动反向传播
            optimizer.step()

        accuracy = evaluate(test_data, net)
        print(f'Epoch {epoch + 1}, Accuracy: {accuracy}')

        rate = rate * 0.8  # 学习率衰减

    # 保存模型
    torch.save(net.state_dict(), 'mnist_model.pth')

if __name__ == "__main__":
    train()