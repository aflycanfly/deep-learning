# -- coding: utf-8 --
# @Time : 8/1/2022 上午 11:38
# @Author : wkq

import torch


def RNN_nolayer():
    # 一共有几句话
    seq_len = 3
    # 抽取几句话
    batch_size = 2
    # 一句话中有几个特征
    input_size = 4

    # 隐层输出几个特征
    hidden_size = 2

    # RNN指定输入的维度，输出的维度
    cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
    # 指定数据等维度
    dataset = torch.randn(seq_len, batch_size, input_size)
    # 指定隐层的维度
    hidden = torch.zeros(batch_size, hidden_size)

    # 枚举类型 循环每次取一句话进行RNN
    for idx, input in enumerate(dataset):
        print("=" * 20, idx, "=" * 20)
        print("Input size", input.shape)

        hidden = cell(input, hidden)

        print("outputs size: ", hidden.shape)
        print(hidden)


def RNN_layer():
    seq_len = 3
    batch_size = 1
    input_size = 4

    hidden_size = 2
    num_layer = 3

    cell = torch.nn.RNN(input_size=input_size, num_layers=num_layer, hidden_size=hidden_size)

    input = torch.randn(seq_len, batch_size, input_size)
    hidden = torch.zeros(num_layer, batch_size, hidden_size)

    out, hidden = cell(input, hidden)
    print("Output_size", out.size())
    print("out", out)
    print("Hidden_size", hidden.size())
    print("Hidden", hidden)


if __name__ == '__main__':
    RNN_nolayer()

    # RNN_layer()
