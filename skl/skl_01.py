# coding: utf-8

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import LinearRegression

f_name = r'/System/Library/Fonts/Hiragino Sans GB W3.ttc'
font = FontProperties(fname=f_name, size=10)

def runplt():
	plt.figure()
	plt.title(u'匹配价格与直径数据', fontproperties = font)
	plt.xlabel(u'直径（英寸）', fontproperties = font)
	plt.ylabel(u'价格（美元）', fontproperties = font)
	plt.axis([0, 25, 0, 25])
	plt.grid(True)
	return plt

# plt = runplt()
X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]
# plt.plot(X, y, 'k.')
# plt.show()
def try_predict():
	model = LinearRegression()
	model.fit(X, y)
	print('预测一张12英寸披萨的价格：$%.2f' % model.predict([12])[0])

plt = runplt()
plt.plot(X, y, 'k.')
X2 = [[0], [10], [14], [25]]
model = LinearRegression()
model.fit(X, y)
y2 = model.predict(X2)
plt.plot(X, y, 'k.')
plt.plot(X2, y2, 'g-')  # 'k': black, 'g':green, '-': solid line style, '.':point marker
plt.show()




