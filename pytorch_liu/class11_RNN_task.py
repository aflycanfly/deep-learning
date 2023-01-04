# -- coding: utf-8 --
# @Time : 8/1/2022 下午 3:00
# @Author : wkq
import torch

'''
hello --> ohlol
'''
seq_len = 5
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
inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
# 等于torch.LongTensor(y_data)
labels = torch.LongTensor(y_data).view(seq_len*batch_size)


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layer=1):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layer = num_layer
        self.rnn = torch.nn.RNN(num_layers=num_layer,input_size=self.input_size, hidden_size=self.hidden_size)

    def forward(self, input):
        hidden = torch.zeros(self.num_layer,self.batch_size, self.hidden_size)
        # self.rnn()返回的是out与hidden
        out,_ = self.rnn(input, hidden)
        # shape(seqLen*batchSize,1)
        return out.view(-1,self.hidden_size)




net = Model(input_size, hidden_size, batch_size)

criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

if __name__ == '__main__':
    for epoch in range(15):
        loss = 0
        # 梯度清零
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _,idx = outputs.max(dim=1)
        idx=idx.data.numpy()
        print("Predicted: ",''.join([idx2char[x] for x in idx]),end='')
        print(',Epoch [%d/15] loss=%0.4f' % (epoch + 1, loss.item()))
