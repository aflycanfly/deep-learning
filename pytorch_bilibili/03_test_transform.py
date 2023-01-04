#-- coding: utf-8 --
#@Time : 12/2/2022 下午 8:09
#@Author : wkq
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


writer = SummaryWriter("logs")
img= Image.open("./resource/hymenoptera_data/train/bees_image/36900412_92b81831ad.jpg")
print(img)
#Totensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("bees_toTensor",img_tensor)

#Nomolize
print("berfore normalize",img_tensor[0][0][0])
trans_normalize = transforms.Normalize([8,9, 5], [0.5, 0.5, 0.5])# 首先看图片有几个通道，第一个列表是mean,第二个是std
img_norm = trans_normalize(img_tensor)
writer.add_image("after normalize Normalize",img_norm,2)
print(img_norm[0][0][0])

# Resize
print("before resize",img_tensor.size())
# 传入的PIL resize后的也是 PIl 出入的是tensor resize后的也是tensor
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img_tensor)
writer.add_image("after resize",img_resize)
print("after resize",img_resize.size())

# Compose - resize -2
trans_compose = transforms.Compose([trans_resize, trans_totensor, trans_normalize])
img_compose = trans_compose(img)
writer.add_image("Compose",img_compose)

# RandomCrop
trans_crop = transforms.RandomCrop(25)
trans_compose2 = transforms.Compose([trans_crop, trans_totensor])
img_crop = trans_compose2(img)
for i in range(10):
    writer.add_image("RandomCrop",img_crop,i)

writer.close()



if __name__ == '__main__':
    print()