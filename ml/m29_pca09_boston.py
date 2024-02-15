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
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=20)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = RandomForestRegressor()
for idx in range(1,len(datasets.feature_names)) :
    
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
'''
[RandomForestRegressor] model r2 :  0.6913310149656464
[RandomForestRegressor] mode eval_r2 :  0.6913310149656464
    pca n_components : 1
    score : 0.7109887285896342
    걸린 시간 : 0.11 초


    pca n_components : 2
    score : 0.7197725814539027
    걸린 시간 : 0.12 초


    pca n_components : 3
    score : 0.7235144361569799
    걸린 시간 : 0.11 초


    pca n_components : 4
    score : 0.6886049027726444
    걸린 시간 : 0.12 초


    pca n_components : 5
    score : 0.7106690330458083
    걸린 시간 : 0.12 초


    pca n_components : 6
    score : 0.7058819154026073
    걸린 시간 : 0.11 초


    pca n_components : 7
    score : 0.7047056243607469
    걸린 시간 : 0.11 초


    pca n_components : 8
    score : 0.7065307003812511
    걸린 시간 : 0.12 초


    pca n_components : 9
    score : 0.7170905618107206
    걸린 시간 : 0.12 초


    pca n_components : 10
    score : 0.6711228438794588
    걸린 시간 : 0.12 초


    pca n_components : 11
    score : 0.7337569945032141
    걸린 시간 : 0.11 초


    pca n_components : 12
    score : 0.7014342633904115
    걸린 시간 : 0.11 초
'''
