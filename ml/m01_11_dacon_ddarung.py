# https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터 - 경로데이터를 메모리에 땡겨옴
path = "c:/_data/dacon/ddarung/"
train_csv = pd.read_csv(path + "train.csv", index_col=0) #c:/_data/dacon/ddarung/train.csv
#문자열에서 경로를 찾아줄때 예약어가 있으면 \를 한번 더 붙여줌.
#경로 찾는것 \ ,\\, /, // 다 됨.
#가독성 면에서 슬래시 모양은 하나로 통일.
#default 맨 윗줄을 컬럼명으로 인식.(대부분 컬럼명이 들어감으로)

test_csv = pd.read_csv(path + "test.csv", index_col=0) 
submission_csv = pd.read_csv(path + "submission.csv")
train_csv = train_csv.fillna(test_csv.mean()) # 715 non-null
test_csv = test_csv.fillna(test_csv.mean()) # 715 non-null

######### x와 y를 분리 ###########
x = train_csv.drop(['count'], axis=1) #axis 0이 행 1이 열
# print(x)

y = train_csv['count'] 
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, train_size=0.7, random_state=12345)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print(x_train.shape) #(1021, 9)

#Early Stopping

#2. 모델
from sklearn.svm import LinearSVR
model = LinearSVR()

#3. 컴파일, 훈련
model.fit(x_train, y_train) 

#4. 평가, 예측
r2 = model.score(x_test, y_test)
y_predict = model.predict(x_test)
r2_pred = r2_score(y_test, y_predict)
print("r2 : ", r2)
print("eval_r2 : ", r2_pred)

y_submit = model.predict(test_csv) # count 값이 예측됨.
submission_csv['count'] = y_submit

######### submission.csv 만들기(count컬럼에 값만 넣어주면됨) ############
import time as tm
ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
file_path = path + f"submission_{save_time}.csv"
submission_csv.to_csv(file_path, index=False)

'''
기존 : 
r2 :  0.5350276904530479
eval_r2 :  0.5350276904530479
'''