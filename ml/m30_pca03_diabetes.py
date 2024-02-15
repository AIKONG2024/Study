from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import numpy as np


# 1. 데이터
datasets = load_diabetes()

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
    pca n_components : 1
    score : 0.1782424208220773
    걸린 시간 : 0.06 초


    pca n_components : 2
    score : 0.37618584388877974
    걸린 시간 : 0.06 초


    pca n_components : 3
    score : 0.37782488614322496
    걸린 시간 : 0.07 초


    pca n_components : 4
    score : 0.5485503452838871
    걸린 시간 : 0.08 초


    pca n_components : 5
    score : 0.5647645448936756
    걸린 시간 : 0.07 초


    pca n_components : 6
    score : 0.5675736419744164
    걸린 시간 : 0.08 초


    pca n_components : 7
    score : 0.5730287691421385
    걸린 시간 : 0.09 초


    pca n_components : 8
    score : 0.5585543643401276
    걸린 시간 : 0.09 초


    pca n_components : 9
    score : 0.5696933664359257
    걸린 시간 : 0.1 초
    
[0.40242108 0.14923197 0.12059663 0.09554764 0.06621814 0.06027171
 0.05365657 0.0433682  0.007832  ]
0.9991439270172946
[0.40242108 0.55165304 0.67224967 0.76779731 0.83401545 0.89428716
 0.94794372 0.99131192 0.99914393]
'''
