import pandas as pd
import matplotlib.pyplot as plt

from kMeansModule import KMeans

# 读取数据
data = pd.read_csv('../data/iris.csv')
iris_types = ['SETOSA', 'VERSICOLOR', 'VIRGINICA']

# 定义横纵坐标名称
x_axis = 'petal_length'
y_axis = 'petal_width'

# figsize=(12, 5)表示图像窗口的大小为 12*5
plt.figure(figsize=(12, 5))

# subplot(1,2,1)表示将整个图像窗口分为1行2列, 当前位置为 1
plt.subplot(1, 2, 1)

# 绘制散点图
for iris_type in iris_types:
    plt.scatter(data[x_axis][data['class'] == iris_type], data[y_axis][data['class'] == iris_type], label=iris_type)
plt.title('label known')
plt.legend()

# subplot(1,2,2)表示将整个图像窗口分为1行2列, 当前位置为 2
plt.subplot(1, 2, 2)
plt.scatter(data[x_axis][:], data[y_axis][:])
plt.title('label unknown')
plt.show()

num_examples = data.shape[0]
x_train = data[[x_axis, y_axis]].values.reshape(num_examples, 2)

# 指定好训练所需的参数
num_clusters = 3
max_iterations = 50

k_means = KMeans(x_train, num_clusters)
centroids, closest_centroids_ids = k_means.train(max_iterations)
# 对比结果
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for iris_type in iris_types:
    plt.scatter(data[x_axis][data['class'] == iris_type], data[y_axis][data['class'] == iris_type], label=iris_type)
plt.title('label known')
plt.legend()

plt.subplot(1, 2, 2)
for centroid_id, centroid in enumerate(centroids):
    current_examples_index = (closest_centroids_ids == centroid_id).flatten()
    plt.scatter(data[x_axis][current_examples_index], data[y_axis][current_examples_index], label=centroid_id)

for centroid_id, centroid in enumerate(centroids):
    plt.scatter(centroid[0], centroid[1], c='black', marker='x')
# legend()用来显示图例,就是图片左上角的标签
plt.legend()

plt.title('label kmeans')
plt.show()
