from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import time

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# 데이터
# x_train, x_test, y_train, y_test = train_test_split(
    # x, y, train_size=0.75, random_state=2874458)

# 모델 구성
model = RandomForestRegressor()
for idx in range(1,len(datasets.feature_names)) :
    x = datasets.data
    y = datasets.target
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    pca = PCA(n_components=idx)
    x = pca.fit_transform(x)
    
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, random_state=2874458)
    
    # 컴파일, 훈련
    model.fit(x_train, y_train)

    # 평가 예측
    start_time = time.time()
    model.fit(x_train, y_train)
    end_time = time.time()
    predict = model.predict(x_test)
    print(f'''
    pca n_components : {idx} 
    score : {r2_score(y_test, predict)}
    걸린 시간 : {round(end_time - start_time ,2 )} 초
    ''')
evr = pca.explained_variance_ratio_
print(evr)
print(evr.sum())

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)

'''
[RandomForestRegressor] model r2 :  0.8106541738490985
[RandomForestRegressor] mode eval_r2 :  0.8106541738490985
    pca n_components : 1 
    score : -0.4580233397999838
    걸린 시간 : 1.78 초


    pca n_components : 2
    score : 0.024783321167045957
    걸린 시간 : 2.48 초


    pca n_components : 3
    score : 0.08225794455896429
    걸린 시간 : 3.02 초


    pca n_components : 4
    score : 0.3287214191088209
    걸린 시간 : 3.41 초


    pca n_components : 5
    score : 0.5937047341234498
    걸린 시간 : 4.03 초


    pca n_components : 6
    score : 0.7014610597786975
    걸린 시간 : 4.57 초


    pca n_components : 7
    score : 0.7676572523724108
    걸린 시간 : 5.09 초

[9.99789327e-01 1.13281110e-04 8.32834638e-05 6.44304641e-06
 5.12871119e-06 2.31833048e-06 1.94839669e-07]
0.9999999762777433
[0.99978933 0.99990261 0.99998589 0.99999233 0.99999746 0.99999978
 0.99999998]
'''

