#-- coding: utf-8 --
#@Time : 30/12/2021 上午 11:00
#@Author : wkq

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# prepare dataset


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]  # shape(多少行，多少列)
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset('../resource/diabetes.csv')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)  # num_workers 多线程


# design model using class


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x



# training cycle forward, backward, update
if __name__ == '__main__':
    model = Model()

    # construct loss and optimizer
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mini_loss=0
    count=0
    epoch_list = []
    loss_list = []

    for epoch in range(100):
        for i, data in enumerate(train_loader, start=0):  # train_loader 是先shuffle后mini_batch
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())

            mini_loss += loss.item()
            count=i

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #     计算每一次迭代的损失
        loss_ava = mini_loss/count
        epoch_list.append(epoch)
        loss_list.append(loss_ava)

    plt.plot(epoch_list, loss_list)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
