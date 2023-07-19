import numpy as np
from utils.features import prepare_for_training


class LinearRegression:

    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
        :param data: 数据集的自变量的数据,也叫特征值
        :param labels: 数据集中的因变量的数据
        :param polynomial_degree: 多项式的阶数
        :param sinusoid_degree: 正弦函数的阶数
        :param normalize_data: 是否需要归一化
        """
        # 1.对数据进行预处理操作
        # 2.先得到所有的特征个数
        # 3.初始化参数矩阵

        # 将四个参数通过prepare_for_training函数进行预处理
        (data_processed, features_mean, features_deviation) = \
            prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data)

        # 将预处理后的数据赋值给类的属性
        self.data = data_processed
        # 保存mean均值
        self.features_mean = features_mean
        # 保存deviation标准差
        self.features_deviation = features_deviation
        # 保存标签用于计算损失,其实就是data的最后一列,也就是y值,是真实值
        self.labels = labels
        # 保存多项式的阶数
        self.polynomial_degree = polynomial_degree
        # 保存正弦函数的阶数
        self.sinusoid_degree = sinusoid_degree
        # 保存是否需要归一化
        self.normalize_data = normalize_data
        # 计算出特征的个数，也就是data的列数
        num_features = self.data.shape[1]
        # 初始化参数矩阵theta,为 列向量
        self.theta = np.zeros((num_features, 1))

    def train(self, alpha, num_iterations=500):
        """
        训练模块，执行梯度下降算法
        """
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iterations):
        """
        执行梯度下降所有操作
        :param alpha:学习率
        :param num_iterations:迭代次数,默认500次
        """
        # cost_history 用于保存每次更新 theta之后的 损失值(也就是 ppt中的目标函数 的值)
        # 如果 损失值 不断减小,说明参数 theta 不断优化,最终会找到 最优 的参数theta
        cost_history = []
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history

    def gradient_step(self, alpha):
        """
        梯度下降参数theta更新的计算方法，注意是矩阵运算
        选择的方法是  小批量梯度下降法
        """

        # 1.批量梯度下降法可以往正确的方向更新参数theta,但是速度很慢
        # 2.随机梯度下降法是随机找一个样本来更新参数theta,速度快,但是不一定往正确的方向更新参数theta
        # 3.通过小批量梯度下降法是选取一部分样本来更新参数theta,速度快,也能往正确的方向更新参数theta,所以我们选择小梯度下降法

        # 样本个数
        num_examples = self.data.shape[0]
        # 预测值
        prediction = self.hypothesis(self.data, self.theta)
        delta = prediction - self.labels
        theta = self.theta

        # 通过小批量梯度下降法不断更新和优化theta,让损失函数的值不断减小,最终找到最优的theta
        theta = theta - alpha * (1 / num_examples) * (np.dot(delta.T, self.data)).T  # np.dot()矩阵乘法
        self.theta = theta

    def cost_function(self, data, labels):
        """
        :param data:数据集的自变量的数据
        :param labels:数据集中的因变量的数据
        :return:损失值
        """
        # 样本个数
        num_examples = data.shape[0]
        # delta为预测值与实际值的差值,为列向量
        delta = self.hypothesis(self.data, self.theta) - labels
        # 目标函数
        cost = (1 / 2) * np.dot(delta.T, delta) / num_examples
        # 计算后的cost大概是长这样  -->  [[23.6]]  ,所以要返回cost[0][0]
        return cost[0][0]

    @staticmethod
    def hypothesis(data, theta):
        """
        :param data: 数据集的自变量的数据
        :param theta: 参数矩阵
        :return:预测值
        """
        # 预测值计算方法
        predictions = np.dot(data, theta)
        return predictions

    def get_cost(self, data, labels):
        """
        用训练的参数模型，与训练集计算损失
        """
        data_processed = \
            prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[0]

        return self.cost_function(data_processed, labels)

    def predict(self, data):
        """
        用训练的参数模型，预测得到回归值结果
        """
        data_processed = \
            prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[0]

        predictions = self.hypothesis(data_processed, self.theta)

        return predictions
