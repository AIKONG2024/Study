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
models = [LinearSVC(), Perceptron(), LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier()]
for model in models:

    #컴파일 , 훈련
    model.fit(x_train, y_train)

    #평가, 예측
    acc = model.score(x_test, y_test)
    y_predict = model.predict(x_test)
    acc_pred = accuracy_score(y_test, y_predict) 
    submission = model.predict(test_csv)
    print(f"[{type(model).__name__}] model acc : ", acc)
    print(f"[{type(model).__name__}] eval_acc : ", acc_pred)
    
    submission_csv['quality'] = submission
    submission_csv['quality'] += 3


    import time as tm
    ltm = tm.localtime(tm.time())
    save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
    file_path = path + f"sampleSubmission{save_time}.csv"
    submission_csv.to_csv(file_path, index=False)
'''
[LinearSVC] model acc :  0.3333333333333333
[LinearSVC] eval_acc :  0.3333333333333333
[Perceptron] model acc :  0.05333333333333334
[Perceptron] eval_acc :  0.05333333333333334
[LogisticRegression] model acc :  0.46545454545454545
[LogisticRegression] eval_acc :  0.46545454545454545
[KNeighborsClassifier] model acc :  0.48727272727272725
[KNeighborsClassifier] eval_acc :  0.48727272727272725
[DecisionTreeClassifier] model acc :  0.576969696969697
[DecisionTreeClassifier] eval_acc :  0.576969696969697
[RandomForestClassifier] model acc :  0.6581818181818182
[RandomForestClassifier] eval_acc :  0.6581818181818182
'''

