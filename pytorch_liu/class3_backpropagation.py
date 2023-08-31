#-- coding: utf-8 --
#@Time : 28/12/2021 下午 10:35
#@Author : wkq

import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0])  # w的初值为1.0
w.requires_grad = True  # 需要计算梯度


def forward(x):
    return x * w  # w是一个Tensor


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

if __name__ == '__main__':
    print("predict (before training)", 4, forward(4).item())

    for epoch in range(100):
        for x, y in zip(x_data, y_data):
            l = loss(x, y)  # l是一个张量，tensor主要是在建立计算图 forward, compute the
            l.backward()  # backward,compute grad for Tensor whose requires_grad set to True
            print('\tgrad:', x, y, w.grad.item())
            w.data = w.data - 0.01 * w.grad.item() # 权重更新时，注意grad也是一个tensor,防止构建计算图
            #只有那些设置了requires_grad=True的张量（tensor）会跟踪其上的所有操作，从而构建一个计算图。这个计算图用于后续的自动微分。
            #如果一个张量的requires_grad属性设置为False（这是默认值），那么在该张量上执行的操作不会被跟踪，因此不会构建计算图。
            w.grad.data.zero_()  # after update, remember set the grad to zero

        print('progress:', epoch, l.item())  # 取出loss使用l.item，不要直接使用l（l是tensor会构建计算图）

    print("predict (after training)", 4, forward(4).item())
