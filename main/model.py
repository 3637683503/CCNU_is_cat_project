import torch
import torch.nn as nn
import torch.nn.init as init


class Net(nn.Module):
    '''
    监控图像的神经网络,判断是否为猫
    '''

    def __init__(self):
        super(Net, self).__init__()
        # 假设输入为彩色图像，输入通道数为3
        self.layer1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=(3, 3))
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
