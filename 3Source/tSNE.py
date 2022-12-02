import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import mfsan

# 对样本进行预处理并画图
def plot_embedding(data, label, flag):
	"""
	:param data:数据集
	:param label:样本标签
	:param title:图像标题
	:return:图像
	"""
	x_min, x_max = np.min(data, 0), np.max(data, 0)
	data = (data - x_min) / (x_max - x_min)		# 对数据进行归一化处理
	# 遍历所有样本
	for i in range(data.shape[0]):
		# 在图中为每个数据点画出标签
		if flag == 0:
			plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(1),
					 fontdict={'weight': 'bold', 'size': 7})
		else:
			plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set2(1),
					 fontdict={'weight': 'bold', 'size': 7})


# 主函数，执行t-SNE降维
def tsne(data, label, flag = 0):
	ts = TSNE(n_components=2, init='pca', random_state=0)
	# t-SNE降维
	reslut = ts.fit_transform(data)
	# 调用函数，绘制图像
	plot_embedding(reslut, label, flag)


# 主函数
if __name__ == '__main__':
	tsne()
