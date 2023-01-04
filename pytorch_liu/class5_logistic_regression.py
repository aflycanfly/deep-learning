#-- coding: utf-8 --
#@Time : 29/12/2021 下午 5:21
#@Author : wkq
import torch

# import torch.nn.functional as F

# prepare dataset
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])


# design model using class
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # y_pred = F.sigmoid(self.linear(x))
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

if __name__ == '__main__':
    model = LogisticRegressionModel()

    # construct loss and optimizer
    # 默认情况下，loss会基于element平均，如果size_average=False的话，loss会被累加。
    criterion = torch.nn.BCELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # training cycle forward, backward, update
    for epoch in range(1000):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        print(epoch, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('w = ', model.linear.weight.item())
    print('b = ', model.linear.bias.item())

    x_test = torch.Tensor([[8.0]])
    y_test = model(x_test)
    print('y_pred = ', y_test.data)