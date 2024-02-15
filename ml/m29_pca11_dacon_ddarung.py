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
import time

#1. 데이터
path = "c:/_data/dacon/ddarung/"
train_csv = pd.read_csv(path + "train.csv", index_col=0) 
test_csv = pd.read_csv(path + "test.csv", index_col=0) 
submission_csv = pd.read_csv(path + "submission.csv")
train_csv = train_csv.fillna(test_csv.mean()) # 715 non-null
test_csv = test_csv.fillna(test_csv.mean()) # 715 non-null

x = train_csv.drop(['count'], axis=1) #axis 0이 행 1이 열
y = train_csv['count'] 

x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, train_size=0.7, random_state=12345)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)
#Early Stopping

#모델 생성
model = RandomForestRegressor()
for idx in range(1,len(train_csv.columns)) :

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

    y_submit = model.predict(test_csv) # count 값이 예측됨.
    submission_csv['count'] = y_submit

    ######### submission.csv 만들기(count컬럼에 값만 넣어주면됨) ############
    import time as tm
    ltm = tm.localtime(tm.time())
    save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}{type(model).__name__}_" 
    file_path = path + f"submission_{save_time}.csv"
    submission_csv.to_csv(file_path, index=False)

'''
[RandomForestRegressor] model r2 :  0.7638923959200988
[RandomForestRegressor] mode eval_r2 :  0.7638923959200988
[RandomForestRegressor] mode eval_mse :  1644.7313593607305

    pca n_components : 1
    score : 0.7620831638085699
    걸린 시간 : 0.23 초


    pca n_components : 2
    score : 0.761283700658139
    걸린 시간 : 0.23 초


    pca n_components : 3
    score : 0.7645813464590047
    걸린 시간 : 0.23 초


    pca n_components : 4
    score : 0.7633502265254889
    걸린 시간 : 0.23 초


    pca n_components : 5
    score : 0.7647329552095843
    걸린 시간 : 0.23 초


    pca n_components : 6
    score : 0.7656531583021234
    걸린 시간 : 0.23 초


    pca n_components : 7
    score : 0.7630800098704419
    걸린 시간 : 0.23 초


    pca n_components : 8
    score : 0.7601318145796851
    걸린 시간 : 0.22 초


    pca n_components : 9
    score : 0.7693383195689083
    걸린 시간 : 0.23 초
'''