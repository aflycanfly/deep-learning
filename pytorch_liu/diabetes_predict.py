#-- coding: utf-8 --
#@Time : 30/12/2021 上午 11:00
#@Author : wkq

import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# 读取原始数据，并划分训练集和测试集
data = pd.read_csv("./diabetes.csv",dtype=np.float32)
raw_data = data.values

X = raw_data[:, 1:]
y = raw_data[:, [0]]
y.resize(y.shape[0])
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3)
Xtest = torch.from_numpy(Xtest)


# 将训练数据集进行批量处理
# prepare dataset

class DiabetesDataset(Dataset):
    def __init__(self, data, label):
        self.len = data.shape[0]  # shape(多少行，多少列)
        self.x_data = torch.from_numpy(data)
        self.y_data = torch.from_numpy(label)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len




train_dataset = DiabetesDataset(Xtrain, Ytrain)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=1)  # num_workers 多线程
test_dataset = DiabetesDataset(Xtrain, Ytrain)
test_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=1)

# design model using class


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)  # 88 = 24x3 + 16

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(512, 10)  # 暂时不知道1408咋能自动出来的

    def forward(self, x):
        in_size = x.size(0)

        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)

        x = x.view(in_size, -1)
        x = self.fc(x)
        return x

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(21, 18)
        self.linear2 = torch.nn.Linear(18, 14)
        self.linear3 = torch.nn.Linear(14, 10)
        self.linear4 = torch.nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return x


model = Model()

# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# training cycle forward, backward, update

def train(epoch):
    train_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, labels = data
        labels = labels.long()
        y_pred = model(inputs)

        loss = criterion(y_pred, labels)
        # print(epoch, i, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, train_loss / 300))
            train_loss = 0.0


    # if epoch % 2000 == 1999:
    #     print("train loss:", train_loss / count, end=',')



def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% ' % (100 * correct / total))

if __name__ == '__main__':

    for epoch in range(10):
        train(epoch)
        test()
