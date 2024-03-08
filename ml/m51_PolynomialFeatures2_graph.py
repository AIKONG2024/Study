import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
plt.rcParams['font.family'] = 'Malgun Gothic'

# 1. 데이터
x = 2 * np.random.rand(100, 1) -1 # -1~1 랜덤 100개 #랜덤값 더하는건 노이즈 추가
y = 3 * x**2 + 2 * x + 1 + np.random.randn(100, 1) # 3x^2 + 2x + 1 + 노이즈 

pf = PolynomialFeatures(degree=2, include_bias= False)
x_poly = pf.fit_transform(x)
# print(x_poly)

#2. 모델
model1 = LinearRegression()
model2 = LinearRegression()

#3. 훈련
model1.fit(x, y)
model2.fit(x_poly, y)

#그래프
plt.scatter(x,y, c = 'blue', label = '원데이터')
plt.xlabel('x')
plt.ylabel('y')
plt.title('polynomal Regression Example')

x_plot = np.linspace(-1,1,100).reshape(-1,1)
x_plot_poly = pf.transform(x_plot)
y_plot = model1.predict(x_plot)
y_plot2 = model2.predict(x_plot_poly)

plt.plot(x_plot, y_plot, color = 'red',label = 'Polynomal Regression')
plt.plot(x_plot, y_plot2, color = 'green',label = '기냥')

plt.legend()
plt.show()