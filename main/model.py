import torch
import torch.nn as nn
import torch.nn.init as init


class Net(nn.Module):
    '''
    识别手写字的神经网络
    '''

    def __init__(self):
        super(Net, self).__init__()
        # 输入层到第一个隐藏层，将28*28 = 784维的输入映射到200维
        self.layer1 = nn.Linear(28 * 28, 200)
        # 第一个隐藏层到第二个隐藏层，将200维映射到100维
        self.layer2 = nn.Linear(200, 100)
        # 第二个隐藏层到第三个隐藏层，将100维映射到40维
        self.layer3 = nn.Linear(100, 40)
        # 第三个隐藏层到输出层，将40维映射到10维，对应10个类别
        self.layer4 = nn.Linear(40, 10)

        # 自定义权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # 将输入数据展平
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.layer1(x))  # 使用ReLU激活函数
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        return self.layer4(x)
