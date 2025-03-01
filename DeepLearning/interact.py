import tkinter as tk
import torch
import numpy as np
from Artificial_idiot import Net


net = Net()
net.load_state_dict(torch.load("mnist_model.pth", weights_only=True))
net.eval()


class GridWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("28x28 格子窗口")

        self.canvas = tk.Canvas(self.root, width=280, height=280)  # 每个格子 10 像素
        self.canvas.pack()

        self.create_grid()  # 创建格子

        self.canvas.bind("<B1 - Motion>", self.draw_on_grid)  # 绑定鼠标拖动事件

        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict_grid)
        self.predict_button.pack(pady=20)

        self.reset_button = tk.Button(self.root, text="Reset", command=self.reset_grid)
        self.reset_button.pack(pady=20)

        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack(pady=20)

        self.root.mainloop()

    def create_grid(self):
        """
        此方法用于创建 28x28 的格子
        """
        self.grid_data = [[0 for _ in range(28)] for _ in range(28)]  # 用于存储每个格子的数据
        for i in range(28):
            for j in range(28):
                x1 = i * 10
                y1 = j * 10
                x2 = x1 + 10
                y2 = y1 + 10
                self.canvas.create_rectangle(x1, y1, x2, y2, fill='white', outline='black')

    def draw_on_grid(self, event):
        """
        此方法用于处理鼠标拖动事件，将经过的格子标记为黑色
        """
        x = event.x // 10
        y = event.y // 10
        if 0 <= x < 28 and 0 <= y < 28:  # 确保在有效格子范围内
            self.grid_data[y][x] = 1
            x1 = x * 10
            y1 = y * 10
            x2 = x1 + 10
            y2 = y1 + 10
            self.canvas.create_rectangle(x1, y1, x2, y2, fill='black', outline='black')

    def predict_grid(self):
        """
        此方法用于对网格中的数据进行预测
        """
        img = np.array(self.grid_data, dtype=np.float32)
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = net(img)
            _, predicted = torch.max(output, 1)
            self.result_label.config(text=f"Predicted digit: {predicted.item()}")

    def reset_grid(self):
        """
        此方法用于重置网格和数据
        """
        self.grid_data = [[0 for _ in range(28)] for _ in range(28)]
        self.canvas.delete("all")
        self.create_grid()


if __name__ == "__main__":
    GridWindow()