import torch

# 定义激活函数（这里使用 Sigmoid 函数）
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# 定义 MLP 类
class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        z1 = self.layer1(x)
        a1 = sigmoid(z1)
        z2 = self.layer2(a1)
        y_pred = sigmoid(z2)
        return y_pred

# 生成一些简单的示例数据（这里使用随机数据）
torch.manual_seed(0)
X = torch.randn(100, 2)  # 100 个样本，每个样本 2 个特征
y = torch.randint(0, 2, (100, 1)).float()  # 二分类标签

# 创建 MLP 实例
mlp = MLP(input_dim=2, hidden_dim=4, output_dim=1)

# 定义损失函数和优化器
loss_func = torch.nn.BCELoss()  # 二元交叉熵损失
optimizer = torch.optim.SGD(mlp.parameters(), lr=0.1)  # 随机梯度下降优化器

# 训练循环
for epoch in range(1000):
    # 前向传播
    y_pred = mlp(X)
    # 计算损失
    loss = loss_func(y_pred, y)
    # 反向传播和更新参数
    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 更新参数

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")