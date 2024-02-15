# https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time

#1. 데이터
path = "c:/_data/dacon/ddarung/"
train_csv = pd.read_csv(path + "train.csv", index_col=0) 
test_csv = pd.read_csv(path + "test.csv", index_col=0) 
submission_csv = pd.read_csv(path + "submission.csv")
train_csv = train_csv.fillna(test_csv.mean()) # 715 non-null
test_csv = test_csv.fillna(test_csv.mean()) # 715 non-null

# x = train_csv.drop(['count'], axis=1) #axis 0이 행 1이 열
# y = train_csv['count'] 

# x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, train_size=0.7, random_state=12345)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)
#Early Stopping

#모델 생성
model = RandomForestRegressor()
for idx in range(1,len(train_csv.columns)) :
    x = train_csv.drop(columns='count', axis=1)
    y = train_csv['count']
    
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    s_test_csv = scaler.transform(test_csv)
    
    pca = PCA(n_components=idx)
    x = pca.fit_transform(x)
    a_test_csv = pca.transform(s_test_csv)
    
    x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, train_size=0.7, random_state=12345)
    #3. 컴파일, 훈련
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

    y_submit = model.predict(a_test_csv) # count 값이 예측됨.
    submission_csv['count'] = y_submit

    ######### submission.csv 만들기(count컬럼에 값만 넣어주면됨) ############
    import time as tm
    ltm = tm.localtime(tm.time())
    save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}{type(model).__name__}_" 
    file_path = path + f"submission_{save_time}.csv"
    submission_csv.to_csv(file_path, index=False)
evr = pca.explained_variance_ratio_
print(evr)
print(evr.sum())

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)

'''
[RandomForestRegressor] model r2 :  0.7638923959200988
[RandomForestRegressor] mode eval_r2 :  0.7638923959200988
[RandomForestRegressor] mode eval_mse :  1644.7313593607305



    pca n_components : 1
    score : -0.22723678450775542
    걸린 시간 : 1.27 초


    pca n_components : 2
    score : 0.2030915520186365
    걸린 시간 : 1.95 초

PS C:\Study>  c:; cd 'c:\Study'; & 'c:\Users\AIA\anaconda3\envs\tf290gpu\python.exe' 'c:\Users\AIA\.vscode\extensions\ms-python.debugpy-2024.0.0-win32-x64\bundled\libs\debugpy\adapter/../..\debugpy\launcher' '53855' '--' 'c:\Study\ml\m30_pca11_dacon_ddarung.py' 

    pca n_components : 1
    score : 0.29512490759598675
    걸린 시간 : 0.1 초


    pca n_components : 2
    score : 0.44821133416421055
    걸린 시간 : 0.12 초


    pca n_components : 3
    score : 0.5447108679799408
    걸린 시간 : 0.15 초


    pca n_components : 4
    score : 0.5802205908200715
    걸린 시간 : 0.17 초


    pca n_components : 5
    score : 0.6429633355410422
    걸린 시간 : 0.19 초


    pca n_components : 6
    score : 0.6424862747891782
    걸린 시간 : 0.21 초


    pca n_components : 7
    score : 0.6600155793697015
    걸린 시간 : 0.23 초


    pca n_components : 8
    score : 0.6765119745262267
    걸린 시간 : 0.26 초


    pca n_components : 9
    score : 0.6696105482638419
    걸린 시간 : 0.28 초

[9.98805418e-01 7.66370837e-04 2.76615404e-04 1.10519162e-04
 2.59310472e-05 1.25827512e-05 2.46472318e-06 9.76012023e-08
 5.02997279e-10]
1.0
[0.99880542 0.99957179 0.9998484  0.99995892 0.99998485 0.99999744
 0.9999999  1.         1.        ]
'''