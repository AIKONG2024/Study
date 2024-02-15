#train_test_split후 스케일링 후 PCA 
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
import sklearn as sk
print(sk.__version__) #1.1.3

#1. 데이터
# datasets = load_diabetes()
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(150, 4) (150,)

scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components=30) #n_components가 1 이면 선하나, 데이터손실이 많아질 수 있음. PCA 하기 전 Standart 스케일링 하는게 정설. 
x = pca.fit_transform(x)
print(x)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=42, shuffle=True
)

#2.모델
model = RandomForestRegressor(random_state=42)
model = RandomForestClassifier(random_state=42)

#3.훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print("=========================")
print(x_train.shape)
print('model.score : ', results)

evr = pca.explained_variance_ratio_
print(evr)
print(evr.sum())

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)

import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
plt.show()

'''
=========================
(120, 4)
model.score :  0.9666666666666667
=========================
(120, 3)
model.score :  0.3
=========================
(120, 2)
model.score :  0.3
=========================
(120, 1)
model.score :  0.3

model.score :  0.4428225673999313
'''