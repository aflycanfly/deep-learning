# -- coding: utf-8 --
# @Time : 12/2/2022 下午 5:29
# @Author : wkq
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import cv2



'''
tensorboard --logdir=code\deep-learning\pytorch_bilibili\logs
'''
#  “logs”保存的文件目录名
writer = SummaryWriter("logs")


# add_scalar测试
def test01():
    r = 5
    x = range(100)
    for i in x:
        # 'y=2x'名称 i*2 y轴 i x轴
        writer.add_scalars('loss', {'loss1': i, 'loss2': i * 2},  i)

    writer.close()


# add_image测试 将图片转换为numpy格式
def test02():
    img_path = "./resource/hymenoptera_data/train/ants_image/7759525_1363d24e88.jpg"
    image = Image.open(img_path)
    image_array = np.array(image)
    writer.add_image(tag="ants2", img_tensor=image_array, global_step=1, dataformats='HWC')


# add_image测试 用cv2读取数据
def test03():
    img_path = "./resource/hymenoptera_data/train/ants_image/7759525_1363d24e88.jpg"
    imread = cv2.imread(img_path)
    writer.add_image(tag="cv2_ants", img_tensor=imread, global_step=1, dataformats='HWC')


if __name__ == '__main__':
    # test01()
    # test02()
    test03()
