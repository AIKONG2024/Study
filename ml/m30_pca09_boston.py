import numpy as np
from sklearn.datasets import load_boston
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import time


datasets= load_boston()
x = datasets.data
y = datasets.target

#데이터 전처리
from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=20)

from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

model = RandomForestRegressor()
for idx in range(1,len(datasets.feature_names)) :
    # 컴파일 훈련
    x = datasets.data
    y = datasets.target
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    pca = PCA(n_components=idx)
    x = pca.fit_transform(x)
    
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.72, random_state=335688) 

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
[RandomForestRegressor] model r2 :  0.6913310149656464
[RandomForestRegressor] mode eval_r2 :  0.6913310149656464
    pca n_components : 1
    score : -0.19547104834578266
    걸린 시간 : 0.06 초


    pca n_components : 2
    score : 0.3808750726639071
    걸린 시간 : 0.07 초


    pca n_components : 3
    score : 0.5555812411531247
    걸린 시간 : 0.07 초


    pca n_components : 4
    score : 0.5991690525370134
    걸린 시간 : 0.08 초


    pca n_components : 5
    score : 0.6398763806373671
    걸린 시간 : 0.09 초


    pca n_components : 6
    score : 0.6516119526874344
    걸린 시간 : 0.09 초


    pca n_components : 7
    score : 0.6868955693689166
    걸린 시간 : 0.1 초


    pca n_components : 8
    score : 0.7164761935605231
    걸린 시간 : 0.11 초


    pca n_components : 9
    score : 0.7172369023054928
    걸린 시간 : 0.11 초


    pca n_components : 10
    score : 0.7507115878961828
    걸린 시간 : 0.12 초


    pca n_components : 11
    score : 0.7488576327915552
    걸린 시간 : 0.12 초


    pca n_components : 12
    score : 0.7548526204974
    걸린 시간 : 0.13 초

[0.47129606 0.11025193 0.0955859  0.06596732 0.06421661 0.05056978
 0.04118124 0.03046902 0.02130333 0.01694137 0.0143088  0.01302331]
0.9951146722737332
[0.47129606 0.581548   0.67713389 0.74310121 0.80731782 0.8578876
 0.89906884 0.92953786 0.9508412  0.96778257 0.98209137 0.99511467]
'''
