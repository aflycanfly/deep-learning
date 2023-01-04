#-- coding: utf-8 --
#@Time : 17/2/2022 下午 3:21
#@Author : wkq


import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1,模型结构+模型参数
torch.save(vgg16, "./save_models/vgg16_method1.pth")

# 保存方式2，模型参数（官方推荐）
torch.save(vgg16.state_dict(), "./save_models/vgg16_method2.pth")

# 陷阱
class Wkq(nn.Module):
    def __init__(self):
        super(Wkq, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

wkq = Wkq()
torch.save(wkq, "./save_models/wkq_method1.pth")