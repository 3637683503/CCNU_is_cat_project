import numpy as np

# 定义激活函数（这里使用 Sigmoid 函数）
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 定义 MLP 类
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # 初始化权重和偏置
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros(output_dim)

    def forward(self, x):
        # 前向传播
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y_pred = sigmoid(self.z2)
        return self.y_pred

    def backward(self, x, y, learning_rate):
        # 反向传播
        m = x.shape[0]
        # 计算输出层的误差
        dz2 = self.y_pred - y
        # 计算输出层权重和偏置的梯度
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0) / m
        # 计算隐藏层的误差
        dz1 = np.dot(dz2, self.W2.T) * self.a1 * (1 - self.a1)
        # 计算隐藏层权重和偏置的梯度
        dW1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0) / m

        # 更新权重和偏置
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

# 生成一些简单的示例数据（这里使用随机数据）
np.random.seed(0)
X = np.random.randn(100, 2)  # 100 个样本，每个样本 2 个特征
y = np.random.randint(0, 2, (100, 1))  # 二分类标签

# 创建 MLP 实例
mlp = MLP(input_dim=2, hidden_dim=4, output_dim=1)

# 训练循环
learning_rate = 0.1
for epoch in range(1000):
    # 前向传播
    y_pred = mlp.forward(X)
    # 计算损失（这里使用二元交叉熵损失）
    loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    # 反向传播和更新参数
    mlp.backward(X, y, learning_rate)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")