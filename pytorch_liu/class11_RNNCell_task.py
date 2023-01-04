# -- coding: utf-8 --
# @Time : 8/1/2022 下午 3:00
# @Author : wkq
import torch

'''
hello --> ohlol
'''
input_size = 4
hidden_size = 4
batch_size = 1

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
# 将x_data数据转换成热编码
x_one_hot = [one_hot_lookup[x] for x in x_data]
# 将热编码转化为input需要的维度 seq_len batch_size input_size
inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
labels = torch.LongTensor(y_data).view(-1, 1)

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)

    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)


net = Model(input_size, hidden_size, batch_size)


criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

if __name__ == '__main__':
    for epoch in range(15):
        loss = 0
        # 梯度清零
        optimizer.zero_grad()
        hidden = net.init_hidden()
        print("Predicted string ",end='')
        # 遍历seq_len次
        for input,label in zip(inputs,labels):
            hidden = net(input,hidden)
            loss += criterion(hidden,label)
            _,idx = hidden.max(dim=1)
            # 输出预测的值
            print(idx2char[idx.item()],end='')

        loss.backward()
        optimizer.step()
        print(',Epoch [%d/15] loss=%0.4f' %(epoch+1,loss.item()))


