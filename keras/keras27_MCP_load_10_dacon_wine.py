# https://dacon.io/competitions/open/235610/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping 
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

#white, red -> 0, 
#OneHotEncoder
# scikit learn 방식
from sklearn.preprocessing import OneHotEncoder
y = y.values.reshape(-1,1) 
one_hot_y = OneHotEncoder(sparse=False).fit_transform(y)

# from keras.utils import to_categoricalsssss
# one_hot_y = to_categorical(y)
# one_hot_y = np.delete(one_hot_y, 0, axis=1)
# print(one_hot_y.shape) 

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
# model = Sequential()
# model.add(Dense(64, input_dim = len(x.columns)))
# model.add(Dense(32))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(7, activation='softmax'))

es = EarlyStopping(monitor='val_loss', mode = 'min', patience= 100, restore_best_weights=True)

#컴파일 , 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# history = model.fit(x_train, y_train, epochs=300, batch_size=10, verbose= 1, validation_split=0.2, callbacks=[es])
model = load_model('..\_data\_save\MCP\keras26_MCP_10_dacon_wine.hdf5')

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
'''