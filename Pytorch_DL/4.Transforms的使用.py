from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms

img_path = "D:\\Project\\McLearning\\Pytorch_DL\\hymenoptera_data\\train\\ants\\0013035.jpg"

# 创建一个 ToTensor的实例对象,并且调用它的 __call__方法
ToTensor_classifier = transforms.ToTensor()
img = Image.open(img_path)
img_tensor = ToTensor_classifier(img)
print(type(img_tensor))  # <class 'torch.Tensor'>

writer = SummaryWriter("logs")
writer.add_image("img_tensor", img_tensor)
# 终端输入 tensorboard --logdir=D:\Project\McLearning\Pytorch_DL\logs

writer.close()
