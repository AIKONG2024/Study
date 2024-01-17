import numpy as np
from sklearn.datasets import load_breast_cancer

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
print(datasets.DESCR) #설명  , 평균 등
print(datasets.feature_names) #컬럼명

x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(569, 30) (569,)
import pandas as pd

#1
unique, counts = np.unique(y, return_counts = True)#[0 1] [212 357]
print(unique, counts)
#2
print(pd.value_counts(y))
print(pd.Series(y).value_counts())
#3
y_df = pd.DataFrame(y)
print(y_df[y_df[0] == 0].count())
print(y_df[y_df[0] == 1].count())


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, random_state=1234)


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#모델 구성
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(64, input_dim = 30)) #기본 함수는 linear
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dropout(0.2))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1,activation='sigmoid')) #이진함수에서는 sigmoid 는 최종레이어에서 들어가야함.

#컴파일 , 훈련
#얼리스토핑
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='auto', 
                   patience=20, restore_best_weights=True, verbose= 1)

import datetime
date = datetime.datetime.now()
print(date) #2024-01-17 10:52:41.770061
date = date.strftime("%m%d_%H%M")
print(date)


mcp_path = '../_data/_save/MCP/cancer/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([mcp_path, 'k26_06_cancer_' ,date, '_', filename]) #체크포인트 가장 좋은 결과들 저장
mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath=filepath)

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy']) #binary_crossentropy 로스지표 이진분류 #훈련 가중치에 반영되는게 아님. 터미널에 각 종류 훈련 loss가 찍힘
                #accuracy == acc 동일하게 사용 가능.,
history = model.fit(x_train, y_train, epochs= 76, batch_size=1, 
          validation_split=0.3,  callbacks=[es, mcp])

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = np.round(model.predict(x_test))
# y_predict = model.predict(x_test)

print(y_predict)

#mse, rmse , rmsle, r2
import numpy as np
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_predict)
print(f"accuracy : {acc}")
print(loss)

history_loss = history.history['loss']
history_val_loss = history.history['val_loss']
history_accuracy = history.history["accuracy"]
import matplotlib.pyplot as plt

plt.rcParams['font.family'] ='Malgun Gothic'
plt.figure(figsize=(9,6))
plt.plot(history_loss, c ='red', label = 'loss', marker = '.')
plt.plot(history_val_loss, c = 'blue', label = 'val_loss', marker = '.')
plt.plot(history_accuracy, c = 'green', label = 'accuracy', marker = '.')
plt.legend(loc = 'upper right')
plt.title('loss accuracy graph')
plt.xlabel = 'epoch'
plt.ylabel = 'loss'
plt.grid()
plt.show()


# # #그래프
# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,6))
# plt.plot(y_predict, c = 'red', label = 'result')
# plt.show()

'''
기존 : 
accuracy : 0.9298245614035088
============================
best : MaxAbs
============================
MinMaxScaler()
 - accuracy : 0.9239766081871345
StandardScaler()
 - accuracy :  0.9473684210526315
MaxAbsScaler()
 - accuracy : 0.9239766081871345
RobustScaler()
 - accuracy :0.9532163742690059
 
 Dropout 적용후:
accuracy : 0.935672514619883
[0.2588944733142853, 0.9356725215911865]
'''