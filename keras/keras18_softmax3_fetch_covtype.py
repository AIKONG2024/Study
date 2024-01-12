from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

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
ohe_y = np.delete(ohe_y,0, axis=1)
print(ohe_y.shape) #(581012, 8) 열 1개가 더 생김
# keras 원핫 인코딩 명령(배열(array)로 변환이됨)
# pandas의 원핫 인코딩은 DataFrame 형태로 변환
# 0부터 라벨링이 주어짐.
print(np.unique(ohe_y, return_counts=True) ) #(array([0., 1.], dtype=float32), array([4067084,  581012], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(x, ohe_y, train_size= 0.7, random_state=1234, stratify=ohe_y)
print(x_test.shape)

#모델 구성
model = Sequential()
model.add(Dense(64, input_dim = 54))
model.add(Dense(64))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))

es = EarlyStopping(monitor='val_loss', mode='min',patience=1000,  verbose=1, restore_best_weights=True)

#컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=80000, batch_size=8000, validation_split=0.2, callbacks=[])

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

arg_y_predict = np.argmax(y_predict, axis=1)
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


