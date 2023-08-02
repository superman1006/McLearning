from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import cv2

# -----------------------------------------------------------------2. writer.add_scalar(tag = 标题, scalar_value = y值, global_step = x值)-----------------------------------------------------------------

# 将logs文件删除，再在终端输入pip cache purge清除缓存，再运行程序
# 创建一个写入器
writer = SummaryWriter("logs")

# 保存模型
for i in range(100):
    # tag:左上角标题
    # scalar_value:x轴的值
    # global_step:y轴的值
    writer.add_scalar(tag='y=3x', scalar_value=3 * i, global_step=i)

# tensorboard读取的是你的logs文件，而不是当前的py文件，所以要先run文件后生成了logs文件才可以在tensorboard中看到图像
# 终端输入 (绝对路径) tensorboard --logdir=D:\Project\McLearning\Pytorch_DL\logs
#      或 (相对路径) tensorboard --logdir=./logs  (可能会出现显示不出来的情况，可以将路径改为绝对路径)

# 出现utf-8问题代表可能你的路径上有中文，将路径改为英文即可
# 出现端口被占用，可以在终端输入netstat -ano查看占用端口的情况,若被占用则可以在上面执行logs语句后面加上--port=8000或其他端口号
# 若tensorboard中的图像和实际不符，这是因为原本写的函数会保存在logs文件当中，当你重新修改了文件的话,tensorboard会进行一个拟合操作，同时呈现了两种情况
# 所以在修改文件后，要将logs文件删除，再重新运行程序


# -----------------------------------------------------------------2. writer.add_image(tag = 标题, img_tensor, global_step,  dataformats='HWC')-----------------------------------------------------------------
# 其中img_tensor要求图片为 torch.Tensor(常用), numpy.array, string, blobname 其中的任意一种类型，所以不可以单独使用 PIL.Image 来读取图片
# 其中的参数dataformats='HWC'表示图片数据的顺序，H代表高度，W代表宽度，C代表通道数，所以dataformats='HWC'表示图片的维度为高*宽*通道数

# 读取图片的方法:
#    1.使用 PIL 和 numpy组合起来读取图片:
#           img_PIL = Image.open(img_path)               <class 'PIL.JpegImagePlugin.JpegImageFile'>
#           img = np.array(img_PIL)                      <class 'numpy.ndarray'>
#    2.使用cv2读取图片:
#           img = cv2.imread(img_path)                   <class 'numpy.ndarray'>
#    3.使用 PIL 和 ToTensor()读取图片:
#           img_PIL = Image.open(img_path)               <class 'PIL.JpegImagePlugin.JpegImageFile'>
#           ToTensor_classifier = transforms.ToTensor()
#           img_tensor = ToTensor_classifier(img_PIL)    <class 'torch.Tensor'>


# ------------------------------------1.使用PIL.Image和numpy组合起来读取图片------------------------------------

# ant图片路径
img_path = "D:\\Project\\McLearning\\Pytorch_DL\\hymenoptera_data\\train\\ants\\0013035.jpg"
img_PIL = Image.open(img_path)
img_np = np.array(img_PIL)
print('img_PIL的类型和shape:', type(img_np), img_np.shape)
writer.add_image("ant", img_np, 1, dataformats='HWC')

# -------------------------------------2.使用cv2读取图片---------------------------------------------------------

# bee图片路径
img_path = "D:\\Project\\McLearning\\Pytorch_DL\\hymenoptera_data\\train\\ants\\0013035.jpg"
img_cv2 = cv2.imread(img_path)
print('img_cv2的类型和shape:', type(img_cv2), img_cv2.shape)

writer.add_image("bee", img_cv2, 1, dataformats='HWC')

# 最后要记得关闭写入器
writer.close()
