# https://dacon.io/competitions/open/235610/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import time


path = "C:/_data/dacon/wine/"

#1.데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

train_csv['type'] = train_csv['type'].replace({"white":1, "red":0})
test_csv['type'] = test_csv['type'].replace({"white":1, "red":0})

x = train_csv.drop(columns='quality')
y = train_csv['quality']

#데이터 분류
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=1234567, stratify=y)

#모델 생성
model = RandomForestClassifier()
for idx in range(1,len(train_csv.columns)) :

    #컴파일 , 훈련
    model.fit(x_train, y_train)

    # 평가 예측
    start_time = time.time()
    model.fit(x_train, y_train)
    end_time = time.time()
    predict = model.predict(x_test)
    print(f'''
    pca n_components : {idx} 
    score : {accuracy_score(y_test, predict)}
    걸린 시간 : {round(end_time - start_time ,2 )} 초
    ''')
    submission = model.predict(test_csv)
    submission_csv['quality'] = submission
    submission_csv['quality'] += 3


    import time as tm
    ltm = tm.localtime(tm.time())
    save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
    file_path = path + f"sampleSubmission{save_time}.csv"
    submission_csv.to_csv(file_path, index=False)
'''
    pca n_components : 1 
    score : 0.6521212121212121
    걸린 시간 : 0.48 초
    

    pca n_components : 2 
    score : 0.6448484848484849
    걸린 시간 : 0.48 초


    pca n_components : 3
    score : 0.6521212121212121
    걸린 시간 : 0.47 초


    pca n_components : 4
    score : 0.6472727272727272
    걸린 시간 : 0.47 초


    pca n_components : 5
    score : 0.6484848484848484
    걸린 시간 : 0.47 초


    pca n_components : 6
    score : 0.6593939393939394
    걸린 시간 : 0.47 초


    pca n_components : 7
    score : 0.6509090909090909
    걸린 시간 : 0.46 초


    pca n_components : 8
    score : 0.6496969696969697
    걸린 시간 : 0.48 초


    pca n_components : 9
    score : 0.6412121212121212
    걸린 시간 : 0.48 초


    pca n_components : 10
    score : 0.6521212121212121
    걸린 시간 : 0.48 초


    pca n_components : 11
    score : 0.6606060606060606
    걸린 시간 : 0.48 초


    pca n_components : 12
    score : 0.6448484848484849
    걸린 시간 : 0.48 초
'''

