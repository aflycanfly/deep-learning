#-- coding: utf-8 --
#@Time : 29/12/2021 下午 5:34
#@Author : wkq

import numpy as np
import torch
import matplotlib.pyplot as plt

# prepare dataset
xy = np.loadtxt('../resource/diabetes.csv', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])  # 第一个‘：’是指读取所有行，第二个‘：’是指从第一列开始，最后一列不要
y_data = torch.from_numpy(xy[:, [-1]])  # [-1] 最后得到的是个矩阵

# design model using class


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)  # 输入数据x的特征是8维，x有8个特征
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()  # 将其看作是网络的一层，而不是简单的函数使用

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))  # y hat
        return x



if __name__ == '__main__':
    model = Model()

    # construct loss and optimizer
    # criterion = torch.nn.BCELoss(size_average = True)
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    epoch_list = []
    loss_list = []
    # training cycle forward, backward, update
    for epoch in range(3000):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        print(epoch, loss.item())
        epoch_list.append(epoch)
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        if epoch % 500 == 499:
            y_pred_label = torch.where(y_pred >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))

            acc = torch.eq(y_pred_label, y_data).sum().item() / y_data.size(0)
            print("loss = ", loss.item(), "acc = ", acc)


    plt.plot(epoch_list, loss_list)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
