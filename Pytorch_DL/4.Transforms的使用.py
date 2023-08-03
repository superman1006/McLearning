from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms

writer = SummaryWriter(log_dir="logs", filename_suffix='_test_transforms')
img_path = "D:\\Project\\McLearning\\Pytorch_DL\\hymenoptera_data\\train\\ants\\0013035.jpg"

# ----------------------------------------------ToTensor()--------------------------------------------------------------
# 创建一个 ToTensor的实例对象,然后调用它的 __call__方法(只接收 PIL / numpy格式)
ToTensor_Instance = transforms.ToTensor()
img = Image.open(img_path)
img_tensor = ToTensor_Instance(img)
print(type(img_tensor))  # <class 'torch.Tensor'>

writer.add_image("img_tensor", img_tensor)
# 终端输入 tensorboard --logdir=D:\Project\McLearning\Pytorch_DL\logs


print('-' * 100)
# ---------------------------------------------Normalize(mean,std)------------------------------------------------------
# 创建一个 Normalize的实例对象,需要传入两个参数,mean中值和std标准差,然后调用它的 __call__方法
# 归一化的过程也就是  (x−E(x)) / σ
print(img_tensor[0][0][0])  # 未归一化的结果
Normalize_Instance = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 3个通道RGB
img_normalize = Normalize_Instance(img_tensor)
writer.add_image("img_normalize", img_normalize)
print(img_normalize[0][0][0])  # 未归一化的结果

print('-' * 100)
# -----------------------------------------------Resize((H,W))----------------------------------------------------------
print(img.size)  # (768, 512)
# 创建一个 Resize的实例对象,需要传入一个参数(H,W),然后调用它的 __call__方法(只接收 PIL 格式!!!)
Resize_Instance = transforms.Resize((512, 512))
img_resize = Resize_Instance(img)
img_resize = ToTensor_Instance(img_resize)
writer.add_image("img_resize", img_resize)

print('-' * 100)
# ----------------------------------------Compose([transforms实例对象])--------------------------------------------------
# Compose是组合的意思,需要传入一个列表[],列表中包含你需要依次执行的操作,这些操作都必须是transforms类型的实例对象,且注意顺序

# 这里代表创建一个Compose的实例对象,列表中传入两个实例对象,Resize和ToTensor,代表先执行Resize,再执行ToTensor
Compose_Instance = transforms.Compose([Resize_Instance, ToTensor_Instance])
img_compose = Compose_Instance(img)
writer.add_image("img_compose", img_compose)

print('-' * 100)
# ------------------------------------------RandomCrop(size)随机裁剪-----------------------------------------------------
RandomCrop_Instance = transforms.RandomCrop(200)
Compose_Instance2 = transforms.Compose([RandomCrop_Instance, ToTensor_Instance])
for i in range(10):
    img_crop = Compose_Instance2(img)
    writer.add_image("img_crops", img_crop, i)

writer.close()
