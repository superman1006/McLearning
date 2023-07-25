"""Normalize features"""

import numpy as np


def normalize(features):
    """
    归一化操作:公式为 (x - mean) / deviation
    :param features: 特征值
    :return: 归一化后的特征值,均值,标准差
    """

    features_normalized = np.copy(features).astype(float)

    # 计算均值
    features_mean = np.mean(features, 0)

    # 计算标准差
    features_deviation = np.std(features, 0)

    # 标准化操作
    if features.shape[0] > 1:
        features_normalized -= features_mean

    # 防止除以0
    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation
