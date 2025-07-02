import torch
import torch.nn as nn
import torch.nn.init as init
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode


class Net(nn.Module):
    '''
    监控图像的神经网络,判断是否为猫
    '''

    def __init__(self):
        super(Net, self).__init__()
        # todo：图像预处理灰度化
        self.layer1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size=(3, 3))
        self.layer2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=(3, 3))
        self.layer3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=(3, 3))
        self.layer4 = nn.Conv2d(in_channels = 64, out_channels = 2, kernel_size=(3, 3))

        # 自定义权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.layer4(x)
        # 全局平均池化
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # 将输出展平为一维向量
        x = x.view(-1, 2)
        return x



def draw_plot(train_losses, train_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Train Accuracy')
    plt.title('Training Accuracy Curve')
    plt.grid(True)

    plt.show()


def evaluate(test_data, net, device):  # 添加 device 参数
    net.eval()  # 设置为评估模式
    n_correct = 0
    n_total = 0

    with torch.no_grad():
        for (x, y) in test_data:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            _, predicted = torch.max(outputs, 1)
            n_correct += (predicted == y).sum()  # 直接累加张量，不调用 item()
            n_total += y.size(0)

    return n_correct.item() / n_total  # 最后只调用一次 item()


def train(device,train_path,test_path):
    '''
    训练模型，并打印出每次的正确率
    '''

    transform = transforms.Compose([
        # 1. 调整大小为(224, 224)，使用双线性插值
        transforms.Resize(
            size=(224, 224),
            interpolation= InterpolationMode.BILINEAR
        ),
        # 2. 转为灰度图（单通道）
        transforms.Grayscale(num_output_channels=1),
        #3.锐化处理
        transforms.RandomAdjustSharpness(sharpness_factor=1) ,
        # 4. 转为张量（自动归一化到[0,1]）
        transforms.ToTensor(),
        # 5. 标准化（使用ImageNet灰度图均值和标准差）
        transforms.Normalize(mean=[0.449], std=[0.226])
        ])

        # 加载训练集
    train_data = DataLoader(
        dataset= datasets.ImageFolder(root=train_path, transform=transform),
        batch_size=15,
        shuffle=True,
        drop_last=True
    )
    # 加载测试集
    test_data = DataLoader(
        dataset=datasets.ImageFolder(root=test_path, transform=transform),
        batch_size=15,
        shuffle=False,
        drop_last=True
    )
    net = Net()
    net = net.to(device)

    print('initial accuracy:', evaluate(test_data, net,device))

    rate = 0.001  # 初始化学习率
    criterion = torch.nn.CrossEntropyLoss()
    train_losses = []
    train_accuracies = []
    for epoch in range(5):  # 假设训练 5 个 epoch
        print(f'赞美欧姆弥撒亚第{epoch}次')
        optimizer = torch.optim.Adam(net.parameters(), lr=rate, weight_decay=1e-3)
        running_loss = 0.0
        for images, labels in train_data:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_data)
        train_losses.append(train_loss)
        accuracy = evaluate(test_data, net,device)
        train_accuracies.append(accuracy)
        rate = rate * 0.8  # 学习率衰减

    # 保存模型
    torch.save(net.state_dict(),'iscat_model.pth')

    return train_losses, train_accuracies


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    train_path = 'path/to/train_data'  #ToDo:更换路径
    test_path = 'path/to/test_data'
    train_losses, train_accuracies = train(device, train_path, test_path)
    draw_plot(train_losses, train_accuracies)