# https://dacon.io/competitions/open/235610/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, LSTM
from keras.callbacks import EarlyStopping , ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

path = "C:/_data/dacon/wine/"

#1.데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

#데이터 확인
print(train_csv.shape)#(5497, 13)
print(test_csv.shape)#(1000, 12)
print(submission_csv.shape)#(1000, 2) "species"

train_csv['type'] = train_csv['type'].replace({"white":1, "red":0})
test_csv['type'] = test_csv['type'].replace({"white":1, "red":0})

x = train_csv.drop(columns='quality')
y = train_csv['quality']

#결측치 확인
print(x.isna().sum())
print(y.isna().sum())

#분류 클래스 확인
print(pd.value_counts(y)) #(array([3, 4, 5, 6, 7, 8, 9], dtype=int64), array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))
print(x.shape)#(5497, 12) 입력값: 12 출력값: 7
print(y.shape)#(5497,)

#OneHotEncoder
# scikit learn 방식
from sklearn.preprocessing import OneHotEncoder
y = y.values.reshape(-1,1) 
one_hot_y = OneHotEncoder(sparse=False).fit_transform(y)

#데이터 분류
x_train, x_test, y_train, y_test = train_test_split(x, one_hot_y, train_size=0.85, random_state=1234567, stratify=one_hot_y)
print(np.unique(y_test, return_counts=True))
#(array([False,  True]), array([9900, 1650], dtype=int64))

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#모델 생성
model = Sequential()
model.add(LSTM(16, input_shape = (12,1)))
model.add(Dense(64, input_dim = len(x.columns)))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(7, activation='softmax'))

es = EarlyStopping(monitor='val_loss', mode = 'min', patience= 100, restore_best_weights=True)
import datetime
date = datetime.datetime.now()
print(date) #2024-01-17 10:52:41.770061
date = date.strftime("%m%d_%H%M")
print(date)


mcp_path = '../_data/_save/MCP/dacon/wine/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([mcp_path, 'k26_10_dacon_wine_' ,date, '_', filename]) #체크포인트 가장 좋은 결과들 저장
mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath=filepath)

#컴파일 , 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=300, batch_size=10, verbose= 1, validation_split=0.2, callbacks=[es, mcp])

#평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스값 : ", loss)
y_predict = model.predict(x_test)
arg_y_test = np.argmax(y_test,axis=1)
arg_y_predict = np.argmax(y_predict, axis=1)

acc_score = accuracy_score(arg_y_test, arg_y_predict) 
print("acc_score :", acc_score)
submission = np.argmax(model.predict(test_csv), axis=1)

submission_csv['quality'] = submission
# print(submission_csv['quality'])
submission_csv['quality'] += 3
# print("+",submission_csv['quality'])


import time as tm
ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
file_path = path + f"sampleSubmission{save_time}.csv"
submission_csv.to_csv(file_path, index=False)

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(history.history['val_loss'], color = 'red', label ='val_loss', marker='.')
plt.plot(history.history['val_acc'], color = 'blue', label ='val_acc', marker='.')
plt.xlabel = 'epochs'
plt.ylabel = 'loss'
plt.show()


'''
기존 : 
loss :  [1.0942447185516357, 0.5296969413757324]
============================
best : MaxAbs
============================
MinMaxScaler()
 - loss : [1.0804797410964966, 0.5490909218788147]
StandardScaler()
 - loss : [1.0863209962844849, 0.5527272820472717]
MaxAbsScaler()
 - loss :  [1.08963942527771, 0.5199999809265137]
RobustScaler()
 - loss :  [1.0902390480041504, 0.54666668176651]
 
 
Dropout() 설정 후 :
로스값 :  [1.0797463655471802, 0.5478788018226624]

CNN ===================
로스값 :  [1.0892804861068726, 0.5381818413734436]
26/26 [==============================] - 0s 660us/step
acc_score : 0.5381818181818182
32/32 [==============================] - 0s 956us/step

RNN ===================
로스값 :  [1.0811997652053833, 0.550303041934967]
26/26 [==============================] - 0s 1ms/step
acc_score : 0.5503030303030303
'''