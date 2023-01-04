# -- coding: utf-8 --
# @Time : 8/1/2022 下午 3:00
# @Author : wkq
import torch

'''
helloe --> ohlolh
'''
seq_len = 6
input_size = 4

num_layer = 2
batch_size = 1
hidden_size = 8
embedding_size = 10

num_class = 4

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3, 0]
y_data = [3, 1, 2, 3, 2, 1]
# 维度 (seq_len,batch)
inputs = torch.LongTensor(x_data).view(seq_len, batch_size)
# 维度 (seq_len*batch)
labels = torch.LongTensor(y_data).view(batch_size * seq_len)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 映射维度是 input_size*embedding_size
        self.emb = torch.nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_size)
        # Input of RNN (seqLen,batchSize,embeddingSize)   Output of RNN (seqLen,batchSize,hiddenSize)
        self.rnn = torch.nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layer)
        # input(N,*,in_features)   output(N,*,out_features)
        self.fc = torch.nn.Linear(hidden_size, num_class)

    def forward(self, x):
        # batch_size = x.size(1)
        hidden = torch.zeros(num_layer, x.size(1), hidden_size)
        # 输入的x维度(seqLen,batchSize,input_size)      -》 输出x的维度(seqLen,batchSize,embeddingSize)
        x = self.emb(x)
        # Input of RNN (seqLen,batchSize,embeddingSize)   Output of RNN (seqLen,batchSize,hiddenSize)  hidden维度（numLayers,batchSize,hiddenSize） self.rnn()返回的是out与hidden
        # hidden 可以理解为最后一个 output 的值
        x, _ = self.rnn(x, hidden)
        # 输入x维度 (seqLen,batchSize,hiddenSize)   输出x维度  (seqLen,batchSize,num_class)
        x = self.fc(x)
        # 返回值维度为（seqLen*batchSize,num_class）
        return x.view(-1, num_class)


net = Model()

criterion = torch.nn.CrossEntropyLoss()
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

        _, idx = outputs.max(dim=1)
        idx = idx.data.numpy()
        print("Predicted: ", ''.join([idx2char[x] for x in idx]), end='')
        print(',Epoch [%d/15] loss=%0.4f' % (epoch + 1, loss.item()))
