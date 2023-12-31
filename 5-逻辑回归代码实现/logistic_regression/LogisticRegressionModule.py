import numpy as np
from scipy.optimize import minimize
from utils.features import prepare_for_training
from utils.hypothesis import sigmoid


class LogisticRegression:
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=False):
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
        # 保存目标值labels,其实就是data的最后一列,也就是y值,是真实值
        self.labels = labels
        # np.unique()函数是去除数组中的重复数字，并进行排序之后输出
        self.unique_labels = np.unique(labels)
        # 保存多项式的阶数
        self.polynomial_degree = polynomial_degree
        # 保存正弦函数的阶数
        self.sinusoid_degree = sinusoid_degree
        # 保存是否需要归一化
        self.normalize_data = normalize_data
        # 计算出特征的个数，也就是data的列数
        num_features = self.data.shape[1]
        # 计算出目标值label的去掉重复值之后的个数,也就是记录这次逻辑回归的结果有几个类别
        num_unique_labels = self.unique_labels.shape[0]
        # 初始化参数矩阵theta,每一次分类的theta都是行向量,都保存在这个矩阵中
        self.theta = np.zeros((num_unique_labels, num_features))

    def train(self, max_iterations=1000):
        """
        训练模块，对每一次分类预处理之后,执行梯度下降算法
        """
        # cost_histories 用于保存每一个分类过程中,每次更新 theta之后的 损失值(也就是 ppt中的目标函数 的值)
        # 如果 损失值 不断减小,说明参数 theta 不断优化,最终会找到 最优 的参数theta
        cost_histories = []
        num_features = self.data.shape[1]

        # 遍历每一次分类
        for unique_label_index, unique_label in enumerate(self.unique_labels):
            # 将该次分类的theta取出来,并且转换成列向量,作为当前分类的初始theta
            current_initial_theta = np.copy(self.theta[unique_label_index].reshape(num_features, 1))

            # 将当前分类的标签值转换成true, false再转化成0, 1的形式,作为当前分类的标签值
            current_lables = (self.labels == unique_label).astype(float)
            (current_theta, cost_history) = \
                self.gradient_descent(self.data, current_lables, current_initial_theta, max_iterations)

            # 当前分类的theta更新完毕之后,将其保存到self.theta中
            self.theta[unique_label_index] = current_theta.T
            cost_histories.append(cost_history)

        return self.theta, cost_histories

    def gradient_descent(self, data, labels, current_initial_theta, max_iterations):
        cost_history = []
        num_features = data.shape[1]
        # 使用scipy中的minimize函数进行梯度下降,得到使得损失函数 minimize 时的的theta,
        # return结果的是一个对象,其中x属性就是theta,fun属性就是损失值,success属性就是是否成功

        result = minimize(
            # 第一个参数,目标函数
            fun=lambda current_theta: self.cost_function(data, labels, current_theta.reshape(num_features, 1)),
            # 第二个参数,最初始的theta值
            x0=current_initial_theta,
            # 第三个参数,选择优化策略,CG代表共轭梯度下降法
            method='CG',
            # 第四个参数,梯度下降迭代计算公式
            jac=lambda current_theta: self.gradient_step(data, labels, current_theta.reshape(num_features, 1)),
            # 第五个参数,每次迭代之后的回调函数,用于保存每次迭代之后的损失值
            callback=lambda current_theta: cost_history.append(
                self.cost_function(data, labels, current_theta.reshape((num_features, 1)))),
            # 第六个参数,最大迭代次数
            options={'maxiter': max_iterations}
        )
        if not result.success:
            raise ArithmeticError('Can not minimize cost function' + result.message)
        optimized_theta = result.x.reshape(num_features, 1)
        return optimized_theta, cost_history

    def gradient_step(self, data, labels, theta):
        """
        梯度下降参数theta更新的计算方法，注意是矩阵运算
        逻辑回归中使用的梯度下降的方法的是随机梯度下降
        """

        # 样本个数
        num_examples = labels.shape[0]
        # 预测值
        predictions = self.hypothesis(data, theta)
        # 求出预测值和真实值之间的差值
        delta = predictions - labels
        theta = (1 / num_examples) * np.dot(data.T, delta)
        # 这里计算后得到的gradients是一个列向量包含所有的特征的梯度值，也就是每一个特征的theta值
        # .flatten()函数是将多维数组降位一维数组,也就是将gradients转换成行向量
        return theta.T.flatten()

    def cost_function(self, data, labels, theta):
        """
        :param data:数据集的自变量的数据
        :param labels:数据集中的因变量的数据
        :param theta:参数矩阵,传入的参数要求是一个列向量
        :return:损失值
        """
        num_examples = data.shape[0]
        predictions = self.hypothesis(data, theta)
        # 逻辑回归的损失函数
        # y_is_one_cost存放的是y=1时的损失值
        # labels[labels == 1]表示的是y=1的所有行
        y_is_one_cost = (-1) * np.dot(labels[labels == 1].T, np.log(predictions[labels == 1]))
        # y_is_zero_cost存放的是y=0时的损失值
        y_is_zero_cost = (-1) * np.dot(1 - labels[labels == 0].T, np.log(1 - predictions[labels == 0]))
        # 计算目标函数,也就是损失函数
        cost = (1 / num_examples) * (y_is_one_cost + y_is_zero_cost)
        return cost

    @staticmethod
    def hypothesis(data, theta):
        """
        :param data: 数据集的自变量的数据
        :param theta: 参数矩阵
        :return:预测值
        """
        predictions = sigmoid(np.dot(data, theta))
        return predictions

    def predict(self, data):
        """
        用训练的参数模型，预测得到回归值结果
        """
        num_examples = data.shape[0]
        data_processed = \
            prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[0]
        prob = self.hypothesis(data_processed, self.theta.T)
        max_prob_index = np.argmax(prob, axis=1)
        class_prediction = np.empty(max_prob_index.shape, dtype=object)
        for index, label in enumerate(self.unique_labels):
            class_prediction[max_prob_index == index] = label
        return class_prediction.reshape((num_examples, 1))
