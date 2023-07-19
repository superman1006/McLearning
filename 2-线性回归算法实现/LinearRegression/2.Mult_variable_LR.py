import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
from LinearRegressionModule import LinearRegression


# ------------------------------------------数据预处理---------------------------------------------
# 通过pandas.read_csv函数读取数据
data = pd.read_csv('../data/world-happiness-report-2017.csv')

# 得到训练和测试数据
train_data = data.sample(frac=0.8)  # 80%的数据作为训练数据
test_data = data.drop(train_data.index)  # 剩下的数据作为测试数据

# 定义自变量参数的名称
input_param_name_1 = 'Economy..GDP.per.Capita.'
input_param_name_2 = 'Freedom'
# 定义因变量参数的名称
output_param_name = 'Happiness.Score'

# 从训练数据中得到自变量和因变量
x_train = train_data[[input_param_name_1, input_param_name_2]].values
y_train = train_data[[output_param_name]].values

# 从测试数据中得到自变量和因变量
x_test = test_data[[input_param_name_1, input_param_name_2]].values
y_test = test_data[[output_param_name]].values

# ----------------------------------------------一, 原始数据 的 绘制-------------------------------------------------------

# ------------------数据------------------
# 创建训练数据的轨迹信息
plot_training_trace = go.Scatter3d(
    x=x_train[:, 0].flatten(),
    y=x_train[:, 1].flatten(),
    z=y_train.flatten(),
    name='Training Set',
    mode='markers',
    marker={
        'size': 10,
        'opacity': 1,
        'line': {
            'color': 'rgb(255, 255, 255)',
            'width': 1
        },
    }
)
# 创建测试数据的轨迹信息
plot_test_trace = go.Scatter3d(
    x=x_test[:, 0].flatten(),
    y=x_test[:, 1].flatten(),
    z=y_test.flatten(),
    name='Test Set',
    mode='markers',
    marker={
        'size': 10,
        'opacity': 1,
        'line': {
            'color': 'rgb(255, 255, 255)',
            'width': 1
        },
    }
)
# 将训练数据和测试数据的轨迹信息放到一起
plot_data = [plot_training_trace, plot_test_trace]
# ------------------布局------------------
# 创建布局的信息
plot_layout = go.Layout(
    title='Date Sets',
    scene={
        # 设置 x y z 坐标轴的标题
        'xaxis': {'title': input_param_name_1},
        'yaxis': {'title': input_param_name_2},
        'zaxis': {'title': output_param_name}
    },
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
)

# ------------------绘制 并 保存为 html------------------
# 传入数据和布局，创建图表
plot_figure = go.Figure(data=plot_data, layout=plot_layout)
plotly.offline.plot(plot_figure, filename='origin_dataset.html', auto_open=False)

# ----------------------------------------------二, 线性回归处理后的 数据 的 绘制--------------------------------------------
# ------------------数据训练------------------
num_iterations = 500
learning_rate = 0.01
polynomial_degree = 0
sinusoid_degree = 0

linear_regression = LinearRegression(x_train, y_train, polynomial_degree, sinusoid_degree)

(theta, cost_history) = linear_regression.train(learning_rate, num_iterations)

print('开始损失', cost_history[0])
print('结束损失', cost_history[-1])

# 绘制损失函数的变化
plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent Progress')
plt.show()

predictions_num = 10

x_min = x_train[:, 0].min()
x_max = x_train[:, 0].max()

y_min = x_train[:, 1].min()
y_max = x_train[:, 1].max()

x_axis = np.linspace(x_min, x_max, predictions_num)
y_axis = np.linspace(y_min, y_max, predictions_num)

x_predictions = np.zeros((predictions_num * predictions_num, 1))
y_predictions = np.zeros((predictions_num * predictions_num, 1))

x_y_index = 0
for x_index, x_value in enumerate(x_axis):
    for y_index, y_value in enumerate(y_axis):
        x_predictions[x_y_index] = x_value
        y_predictions[x_y_index] = y_value
        x_y_index += 1

z_predictions = linear_regression.predict(np.hstack((x_predictions, y_predictions)))

# 创建预测数据的轨迹信息
plot_predictions_trace = go.Scatter3d(
    x=x_predictions.flatten(),
    y=y_predictions.flatten(),
    z=z_predictions.flatten(),
    name='Prediction Plane',
    mode='markers',
    marker={
        'size': 1,
    },
    opacity=0.8,
    surfaceaxis=2,
)
# 将 第一步处理好的训练数据和测试数据 以及 第二部处理好的预测的轨迹信息放到一起
plot_data = [plot_training_trace, plot_test_trace, plot_predictions_trace]
# ------------------绘制 并 保存为 html------------------
# 传入数据和布局，创建图表
plot_figure = go.Figure(data=plot_data, layout=plot_layout)
plotly.offline.plot(plot_figure, filename='processed_dataset.html', auto_open=False)
