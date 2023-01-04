# -- coding: utf-8 --
# @Time : 13/2/2022 下午 1:05
# @Author : wkq
import os
from PIL import Image
import torchvision

# 准备的测试数据集
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


class MyData(Dataset):
    def __init__(self, root_dir, lable_dir, transform=None):
        self.root_dir = root_dir
        self.lable_dir = lable_dir
        self.path = os.path.join(root_dir, lable_dir)
        # 列出当前目录path下的所有文件名
        self.img_names = os.listdir(self.path)
        self.transform = transform

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_item_path = os.path.join(self.root_dir, self.lable_dir, img_name)
        img = Image.open(img_item_path)
        if self.transform is not None:
            img = self.transform(img)
        lable = self.lable_dir
        return img, lable

    def __len__(self):
        return len(self.img_names)


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((512,512)),
    torchvision.transforms.ToTensor()
])
ants_data = MyData(root_dir="./resource/hymenoptera_data/train", lable_dir="ants_image", transform=transform)

ants_loader = DataLoader(dataset=ants_data, batch_size=4, shuffle=True, num_workers=0, drop_last=True)

# 测试数据集中第一张图片及target
img, target = ants_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
# shuffle=True 两轮数据不一致
for epoch in range(2):
    step = 0
    for data in ants_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step = step + 1

writer.close()
