from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, LSTM
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint

datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(pd.value_counts(y))

# print(x.shape, y.shape) #(581012, 54) (581012,)
# print(np.unique(y, return_counts=True))
# print(pd.value_counts(y))

#결측치 확인
# print(pd.isna(x).sum())
# print(pd.isna(y).sum())

#One hot Encoder

#pandas
# ohe_y = pd.get_dummies(y)
# print(ohe_y.shape)

# scikit learn
y = y.reshape(-1,1)
ohe_y = OneHotEncoder(sparse=False).fit_transform(y)
print(ohe_y.shape)#(581012, 7)
print(np.unique(ohe_y, return_counts=True) ) #(array([0., 1.]), array([3486072,  581012], dtype=int64))
pd_y = pd.DataFrame(ohe_y)
print(pd_y.columns)

#keras
ohe_y = to_categorical(y)
# ohe_y = np.delete(ohe_y,0, axis=1)
ohe_y = ohe_y[:,1:]
print(ohe_y.shape) #(581012, 8) 열 1개가 더 생김
# keras 원핫 인코딩 명령(배열(array)로 변환이됨)
# pandas의 원핫 인코딩은 DataFrame 형태로 변환
print(np.unique(ohe_y, return_counts=True) ) #(array([0., 1.], dtype=float32), array([4067084,  581012], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(x, ohe_y, train_size= 0.7, random_state=1234, stratify=ohe_y)
print(x_test.shape)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)
print(x_train.shape)

x_train = x_train.reshape(-1,54,1,1)
x_test = x_test.reshape(-1,54,1,1)


#모델 구성
model = Sequential()
model.add(LSTM(16, input_shape = (54,1)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))
#0.6371110244171103
es = EarlyStopping(monitor='val_loss', mode='min',patience=1000,  verbose=1, restore_best_weights=True)
import datetime
date = datetime.datetime.now()
print(date) #2024-01-17 10:52:41.770061
date = date.strftime("%m%d_%H%M")
print(date)


mcp_path = '../_data/_save/MCP/fetch_covtyp/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([mcp_path, 'k26_09_fetch_covtyp_' ,date, '_', filename]) #체크포인트 가장 좋은 결과들 저장
mcp = ModelCheckpoint(monitor='val_loss', mode= 'min', verbose=1, save_best_only= True, filepath=filepath)

#컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=1000, batch_size=8000, validation_split=0.2, callbacks=[es, mcp])

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

arg_y_predict = np.argmax(y_predict, axis=1)
print("=" * 100)
print(pd.value_counts(arg_y_predict))

arg_y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(arg_y_predict,arg_y_test)
print("loss : ", loss)
print("acc_score : ", acc_score)

#시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(history.history['val_loss'], color = 'red', label ='val_loss', marker='.')
plt.plot(history.history['val_acc'], color = 'blue', label ='val_acc', marker='.')
plt.xlabel = 'epochs'
plt.ylabel = 'loss'
plt.show()


'''
기존 : 
loss :  loss :  [0.5527498126029968, 0.7625470161437988]
============================
best : MaxAbs
============================
MinMaxScaler()
 - loss : [0.45974233746528625, 0.8049327731132507]
StandardScaler()
 - loss :  [0.45520448684692383, 0.8062637448310852]
MaxAbsScaler()
 - loss :  [0.4693068563938141, 0.8011634945869446]
RobustScaler()
 - loss :  [0.42321649193763733, 0.8228898644447327]
 
 
Dropout() 적용후:
loss :  [0.5161565542221069, 0.77559894323349]

CNN ====================================
loss :  [0.5168811678886414, 0.7762702107429504]
acc_score :  0.7762701946025335

RNN ====================================
loss :  [0.6018034815788269, 0.7396273016929626]
acc_score :  0.739627317789609

'''