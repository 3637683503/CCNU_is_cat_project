import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model import Net
import matplotlib.pyplot as plt


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


def evaluate(test_data, net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            x = x.to(device)
            y = y.to(device)
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
        dataset=datasets.MNIST('', train=True, transform=transform, download=True),
        batch_size=15,
        shuffle=True,
        drop_last=True
    )
    # 加载测试集
    test_data = DataLoader(
        dataset=datasets.MNIST('', train=False, transform=transform, download=True),
        batch_size=15,
        shuffle=False,
        drop_last=True
    )
    net = Net()
    net = net.to(device)

    print('initial accuracy:', evaluate(test_data, net))

    rate = 0.001  # 初始化学习率
    criterion = torch.nn.CrossEntropyLoss()
    train_losses = []
    train_accuracies = []
    for epoch in range(5):  # 假设训练 5 个 epoch
        print('赞美欧姆弥撒亚')
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
        accuracy = evaluate(test_data, net)
        train_accuracies.append(accuracy)
        rate = rate * 0.8  # 学习率衰减

    # 保存模型
    torch.save(net.state_dict(),'mnist_model.pth')

    return train_losses, train_accuracies


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    train_losses, train_accuracies = train()
    draw_plot(train_losses, train_accuracies)