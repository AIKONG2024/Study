# https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error #mse

#1. 데이터 - 경로데이터를 메모리에 땡겨옴
path = "c:/_data/dacon/ddarung/"
train_csv = pd.read_csv(path + "train.csv", index_col=0) #c:/_data/dacon/ddarung/train.csv
#문자열에서 경로를 찾아줄때 예약어가 있으면 \를 한번 더 붙여줌.
#경로 찾는것 \ ,\\, /, // 다 됨.
#가독성 면에서 슬래시 모양은 하나로 통일.
#default 맨 윗줄을 컬럼명으로 인식.(대부분 컬럼명이 들어감으로)

# print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col=0) 
# print(test_csv)
submission_csv = pd.read_csv(path + "submission.csv")
# print(submission_csv)

# print(train_csv.shape)# (1459, 10)
# print(test_csv.shape)# (715, 9)
# print(submission_csv.shape)# (715, 2) #test_csv 와 submission_csv 의 열 합이 12개인이유: id 컬럼이 중복임.

# print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')

# print(train_csv.info())
# print(test_csv.info())
# print(train_csv.describe())#데이터 숫자, 평균값, 표준편차....25% (1분위) 등등...

######### 결측치 처리 1. 제거 #########
# print(train_csv.isnull().sum())
# print(train_csv.isna().sum()) #isna() == isnull()
# train_csv = train_csv.dropna() #결측치가 속한 행 전부 삭제됨.
# train_csv = train_csv.fillna(0)
# train_csv['hour_bef_precipitation'].fillna(value= 0.0 , inplace=True)
train_csv = train_csv.fillna(test_csv.mean()) # 715 non-null

# print(train_csv.isna().sum()) #isna() == isnull()
# print(train_csv.info())
# print(train_csv.shape)  #(1328, 10)

# test_csv['hour_bef_precipitation'].fillna(value= 0.0 , inplace=True)
test_csv = test_csv.fillna(test_csv.mean()) # 715 non-null
# test_csv = test_csv.fillna(0) # 715 non-null

# print(test_csv.info())

######### x와 y를 분리 ###########
x = train_csv.drop(['count'], axis=1) #axis 0이 행 1이 열
# print(x)

y = train_csv['count'] 
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, train_size=0.8, random_state=341)

# print(x_train.shape, x_test.shape) #(929, 9) (399, 9)
# print(y_train.shape, y_test.shape) #(929,) (399,)

#2. 모델
model = Sequential()
model.add(Dense(64, input_dim = len(x.columns)))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=150, validation_split=0.3)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("loss : ",loss)
print("r2 : ", r2)

y_submit = model.predict(test_csv) # count 값이 예측됨.
submission_csv['count'] = y_submit

######### submission.csv 만들기(count컬럼에 값만 넣어주면됨) ############
import time as tm
ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
file_path = path + f"submission_{save_time}.csv"
submission_csv.to_csv(file_path, index=False)

import matplotlib.pyplot as plt
plt.scatter(y_test, y_predict)
plt.plot(y_submit, color = 'red')  # Plotting actual values
plt.show()


#train_size : 0.8 /  deep 6 (9-128-64-1)  / random_state : 6131483 / epochs = 340 / batch_size = 32
#train mean test mean 
# loss :  2536.41650390625
# r2 :  0.6300840825839278


#validation_data 적용전
# loss :  1940.718505859375
# r2 :  0.6266871970848431

#적용후
# loss :  2147.240478515625
# r2 :  0.6396772014593524