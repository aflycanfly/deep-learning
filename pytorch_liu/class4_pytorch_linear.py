#-- coding: utf-8 --
#@Time : 29/12/2021 下午 3:27
#@Author : wkq


import torch

# 1.准备数据集
# x,y是矩阵，3行1列 也就是说总共有3个数据，每个数据只有1个特征
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

# 2.设计模型
"""
our model class should be inherit from nn.Module, which is base class for all neural network modules.
member methods __init__() and forward() have to be implemented
class nn.linear contain two member Tensors: weight and bias
class nn.Linear has implemented the magic method __call__(),which enable the instance of the class can
be called just like a function.Normally the forward() will be called 
"""
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # (1,1)是指输入x和输出y的特征维度，这里数据集中的x和y的特征都是1维的
        # 该线性层需要学习的参数是w和b  获取w/b的方式分别是~linear.weight/linear.bias
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

if __name__ == '__main__':
    model = LinearModel()

    # 3.构建损失和优化器
    # criterion = torch.nn.MSELoss(size_average = False)
    criterion = torch.nn.MSELoss(reduction='sum',size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # model.parameters()自动完成参数的初始化操作

    # 4.训练周期向前、向后、更新
    for epoch in range(100):
        y_pred = model(x_data)  # forward:predict
        loss = criterion(y_pred, y_data)  # forward: loss
        print(epoch, loss.item())

        optimizer.zero_grad()  # the grad computer by .backward() will be accumulated. so before backward, remember set the grad to zero
        loss.backward()  # backward: autograd，自动计算梯度
        optimizer.step()  # update 参数，即更新w和b的值

    print('w = ', model.linear.weight.item())
    print('b = ', model.linear.bias.item())

    x_test = torch.tensor([[4.0]])
    y_test = model(x_test)
    print('y_pred = ', y_test.data)