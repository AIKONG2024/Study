from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense , Dropout, Conv2D, Flatten, LSTM
from keras.utils import to_categorical  
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping, ModelCheckpoint

datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(178, 13) (178,)
print(pd.value_counts(y)) 
# np.unique(y, return_counts=True)
'''
1    71
0    59
2    48
'''

#one hot encoder
#1. keras
ohe_y = to_categorical(y)
#2. pandas
ohe_y = pd.get_dummies(y)
#3. scikitlearn
y = y.reshape(-1,1)
ohe_y = OneHotEncoder(sparse=False).fit_transform(y)

print(ohe_y)
print(ohe_y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, ohe_y, train_size= 0.72, random_state=123, stratify=ohe_y)
print(np.unique(y_train, return_counts=True))

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#모델 구현
model = Sequential()
model.add(LSTM(16,input_shape = (13,1)))
model.add(Dense(64, input_dim = 13))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(3, activation='softmax'))

es = EarlyStopping(monitor='val_loss', mode='min', patience=80, verbose=1, restore_best_weights=True)
import datetime
date = datetime.datetime.now()
print(date) #2024-01-17 10:52:41.770061
date = date.strftime("%m%d_%H%M")
print(date)


mcp_path = '../_data/_save/MCP/wine/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([mcp_path, 'k26_08_wine_' ,date, '_', filename]) #체크포인트 가장 좋은 결과들 저장
mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath=filepath)

#컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=550, batch_size=1, validation_split=0.2, verbose=1,callbacks=[es, mcp])

#예측 평가
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_test)
print(y_predict)

arg_y_test = np.argmax(y_test, axis=1)
arg_y_predict = np.argmax(y_predict, axis=1)

acc_score = accuracy_score(arg_y_test, arg_y_predict)
print("로스 : ", loss[0])
print("정확도 : ", loss[1])

print("acc score :", acc_score)

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(history.history['val_acc'], color = 'blue', label = 'val_acc', marker = '.')
plt.plot(history.history['val_loss'], color = 'red', label = 'val_loss', marker = '.')
plt.show()

'''
기존 : 
accuracy 0.8999999761581421
============================
best : MaxAbs
============================
MinMaxScaler()
정확도 :  0.9599999785423279
StandardScaler()
정확도 :  0.8999999761581421
MaxAbsScaler()
 - accuracy : 0.9239766081871345
RobustScaler()
 - accuracy :0.9532163742690059
 
Dropout() 적용후:
 정확도 :  0.9200000166893005
 
CNN ========================
로스 :  0.006823524367064238
정확도 :  1.0
acc score : 1.0
RNN ========================
로스 :  0.16561217606067657
정확도 :  0.9599999785423279
acc score : 0.96
'''