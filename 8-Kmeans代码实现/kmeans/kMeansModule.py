import numpy as np


class KMeans:
    def __init__(self, data, num_clusters):
        self.data = data
        self.num_clusters = num_clusters

    def train(self, max_iterations):
        """
        训练函数
        1.先随机选择 K 个中心点(不是在data中随机选择,就是在坐标中随机选择)
            { 2.计算每一个样本点到 K 个中心点的距离                          }
            { 3.标记每个样本点的类别(每个样本距离那个中心点最近, 就属于那个类别)  }
            { 4.计算每个类别中的数据到对应类别中心点的距,进行中心点位置更新      }
        5.重复2-4步骤,直到中心点位置不再变化 或者 达到最大迭代次数
        """
        # 1.先随机选择K个中心点
        centPoint = self.centPoint_init(self.data, self.num_clusters)
        # 2.开始训练
        num_examples = self.data.shape[0]
        # np.empty()函数用来创建一个指定形状（shape）、数据类型（dtype）且未初始化的数组
        closest_centPoint_ids = np.empty((num_examples, 1))
        for _ in range(max_iterations):
            # 3得到当前每一个样本点到K个中心点的距离，找到最近的
            closest_centPoint_ids = self.centPoint_find_closest(self.data, centPoint)
            # 4.进行中心点位置更新
            centPoint = self.centPoint_compute(self.data, closest_centPoint_ids, self.num_clusters)
        return centPoint, closest_centPoint_ids

    @staticmethod
    def centPoint_init(data, num_clusters):
        num_examples = data.shape[0]
        # np.random.permutation()函数的作用是打乱原来的序列，使原来序列的每个元素以相同的概率出现在新序列中
        random_ids = np.random.permutation(num_examples)
        # 从打乱的序列中取出前num_clusters个作为中心点,后面的“ , : ”代表取所有的列
        centPoint = data[random_ids[:num_clusters], :]
        return centPoint

    @staticmethod
    def centPoint_find_closest(data, centPoint):
        num_examples = data.shape[0]
        num_centPoint = centPoint.shape[0]
        closest_centPoint_ids = np.zeros((num_examples, 1))
        for example_index in range(num_examples):
            distance = np.zeros((num_centPoint, 1))
            for centPoint_index in range(num_centPoint):
                distance_diff = data[example_index, :] - centPoint[centPoint_index, :]
                distance[centPoint_index] = np.sum(distance_diff ** 2)
            closest_centPoint_ids[example_index] = np.argmin(distance)
        return closest_centPoint_ids

    @staticmethod
    def centPoint_compute(data, closest_centPoint_ids, num_clusters):
        num_features = data.shape[1]
        centPoint = np.zeros((num_clusters, num_features))
        for centPoint_id in range(num_clusters):
            closest_ids = closest_centPoint_ids == centPoint_id
            centPoint[centPoint_id] = np.mean(data[closest_ids.flatten(), :], axis=0)
        return centPoint
