#-- coding: utf-8 --
#@Time : 12/2/2022 下午 2:17
#@Author : wkq
from PIL import  Image
from torch.utils.data import Dataset
import os
'''

将ants_lable目录中的用TXT文件，存储每个图片的类别
import  os
root_dir = "./resource/hymenoptera_data/train"
target_dir = "ants_image"
img_names = os.listdir(os.path.join(root_dir,target_dir))
lable = target_dir.split('_')[0]
out_dir = "ants_label"
for i in img_names:
    file_name = i.split(".jpg")[0]
    with open(os.path.join(root_dir,out_dir,"{}.txt".format(file_name)),'w') as f:
        f.write(lable)

'''

class MyData(Dataset):
    def __init__(self,root_dir,lable_dir):
        self.root_dir = root_dir
        self.lable_dir = lable_dir
        self.path  = os.path.join(root_dir,lable_dir)
        # 列出当前目录path下的所有文件名
        self.img_names = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_item_path = os.path.join(self.root_dir, self.lable_dir, img_name)
        img = Image.open(img_item_path)
        lable = self.lable_dir
        return img,lable
    def __len__(self):
        return len(self.img_names)

root_dir = "./resource/hymenoptera_data/train"
ants_lable_dir = "ants_image"
bees_lable_dir = "bees_image"
ants_dataset = MyData(root_dir=root_dir,lable_dir=ants_lable_dir)
bees_dataset = MyData(root_dir, bees_lable_dir)

if __name__ == '__main__':
    print()