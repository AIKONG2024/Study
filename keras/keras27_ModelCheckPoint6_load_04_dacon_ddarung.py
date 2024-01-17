

# # https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
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

x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, train_size=0.7, random_state=12345)

# print(x_train.shape, x_test.shape) #(929, 9) (399, 9)
# print(y_train.shape, y_test.shape) #(929,) (399,)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#Early Stopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=3000, verbose=1,  restore_best_weights=True)

#2. 모델
# model = Sequential()
# model.add(Dense(512, input_dim = len(x.columns)))
# model.add(Dense(256))
# model.add(Dense(128))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))
#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'acc'])
# hist = model.fit(x_train, y_train, epochs=300000, batch_size=70, validation_split=0.3, callbacks=[es]) #98
path = '../_data/_save/MCP/dacon/ddarung/'
model = load_model('k26_0117_1215_0104-26.9510.hdf5')

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

#시각화
# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] ='Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] =False
# history_loss = hist.history['loss']
# history_val_loss = hist.history['val_loss']
# plt.figure(figsize=(9,6)) #세로 9 가로 6
# plt.plot(history_loss, c = 'red', label = 'loss', marker = '.' )
# plt.plot(history_val_loss, c = 'blue', label = 'val_loss', marker = '.' )
# plt.legend(loc = 'upper right')
# plt.title('따르릉 찻트')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()
# plt.show()






'''
기존 : 
loss :  [2606.534423828125, 2606.534423828125, 35.9389533996582, 0.0022831049282103777]
============================
best : MaxAbs
============================
MinMaxScaler()
 - loss : [2618.3544921875, 2618.3544921875, 33.959373474121094, 0.004566209856420755]
StandardScaler()
 - loss :  [2469.43359375, 2469.43359375, 33.88142013549805, 0.004566209856420755]
MaxAbsScaler()
 - loss :  [2664.299072265625, 2664.299072265625, 35.0896110534668, 0.004566209856420755]
RobustScaler()
 - loss : [2447.616943359375, 2447.616943359375, 33.25836181640625, 0.0022831049282103777]
 
 standard : [1890.7735595703125, 1890.7735595703125, 29.455768585205078, 0.004566209856420755]
'''