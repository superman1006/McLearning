import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LinearRegressionModule import LinearRegression

# 通过pandas.read_csv函数读取数据
data = pd.read_csv('../data/world-happiness-report-2017.csv')

# 得到训练和测试数据
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

# 定义自变量参数的名称
input_param_name = 'Economy..GDP.per.Capita.'
# 定义因变量参数的名称
output_param_name = 'Happiness.Score'

# 从训练数据中得到自变量和因变量
x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values

# 从测试数据中得到自变量和因变量
x_test = test_data[input_param_name].values
y_test = test_data[output_param_name].values

# 绘制 训练数据 和 测试数据 的散点(scatter)图
plt.scatter(x_train, y_train, label='Train data')
plt.scatter(x_test, y_test, label='test data')
# 定义x轴和y轴的名称
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
# 定义图的标题
plt.title('Happy')
# 定义图例
plt.legend()
plt.show()

# 定义 学习率alpha 和 迭代次数
learning_rate = 0.01
num_iterations = 500

# 创建线性回归模型实例对象
linear_regression = LinearRegression(x_train, y_train)
(theta, cost_history) = linear_regression.train(learning_rate, num_iterations)

print('theta: ', theta)
print('开始时的损失值:', cost_history[0])
print('训练后的损失值:', cost_history[-1])

plt.plot(range(num_iterations), cost_history)
plt.xlabel('num_iterations')
plt.ylabel('cost')
plt.title('GD')
plt.show()

predictions_num = 100
# 通过np.linspace(起始值, 结束值, 点的个数)来自动生成一个等差数列的数组
# reshape(A, B) 将数组转换成一个 A行 B列 的数组
x_predictions = np.linspace(x_train.min(), x_train.max(), predictions_num).reshape(predictions_num, 1)
print('x_predictions: ', x_predictions)
y_predictions = linear_regression.predict(x_predictions)

plt.scatter(x_train, y_train, label='Train data')
plt.scatter(x_test, y_test, label='test data')
# 绘制预测的直线
plt.plot(x_predictions, y_predictions, 'r', label='Prediction')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()
plt.show()
