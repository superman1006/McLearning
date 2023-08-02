from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# train数据集、val数据集、test数据集区别: https://blog.csdn.net/kupepoem/article/details/101055179
# 将数据集放入目录中要用复制粘贴 ,不要直接拖动
# 数据的地址 不能用单斜杠\ ,要用 双斜杠 \\ 代表转义

# 单个图片的地址
img_path = "D:\\Project\\机器学习\\Pytorch_DL\\hymenoptera_data\\train\\ants\\0013035.jpg"
img = Image.open(img_path)
print("img.size: ", img.size)

# ant 训练的数据集的地址
dataset_path = "D:\\Project\\机器学习\\Pytorch_DL\\hymenoptera_data\\train"
label = "ants"
# os.path.join() 方法用于将多个路径组合后返回,在这里代表将dataset_path和label组合得到ants数据集的地址
data_path = os.path.join(dataset_path, label)
# os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
data_path_list = os.listdir(data_path)

print('-' * 100)


class MyDataset(Dataset):  # 继承抽象类Dataset

    def __init__(self, root_dir: str, label_dir: str):
        """
        :param root_dir: 数据集的根目录
        :param label_dir: 数据集的标签
        """
        self.root_dir = root_dir
        self.label_dir = label_dir
        # 上述两者组合得到数据集的地址
        self.path = os.path.join(self.root_dir, self.label_dir)
        # 获取数据集的所有图片的名字
        self.img_name_list = os.listdir(self.path)

    def __getitem__(self, index):
        """
        :param index: 数据集的索引
        :return: 返回对应index的图片和标签
        """
        # 获取对应index的图片的名字
        img_name = self.img_name_list[index]
        # 获取对应index的图片的地址
        img_item_path = os.path.join(self.path, img_name)
        # 打开对应index的图片
        img = Image.open(img_item_path)
        return img, self.label_dir

    def __len__(self):
        return len(self.img_name_list)


if __name__ == '__main__':
    root_dir = "D:\\Project\\机器学习\\Pytorch_DL\\hymenoptera_data\\train"
    label_dir = "ants"
    # 创建数据集
    ants_dataset = MyDataset(root_dir, label_dir)
    print(len(ants_dataset.img_name_list))
    img, label = ants_dataset[12]
    print(img, label)
    img.show()
