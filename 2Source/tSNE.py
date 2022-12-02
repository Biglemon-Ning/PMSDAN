import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import mfsan

# 对样本进行预处理并画图
def plot_embedding(data, label, flag, subplots):
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
		if i <= flag:
			subplots.text(data[i, 0], data[i, 1], str(label[i]), color='red',
					 fontdict={'weight': 'bold', 'size': 12})
		else:
			subplots.text(data[i, 0], data[i, 1], str(label[i]), color='deepskyblue',
					 fontdict={'weight': 'bold', 'size': 12})


# 主函数，执行t-SNE降维
def tsne(data, label, sub, flag = 0):
	ts = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
	# t-SNE降维
	reslut = ts.fit_transform(data)
	# 调用函数，绘制图像
	plot_embedding(reslut, label, flag, sub)


# 主函数
if __name__ == '__main__':
	tsne()
