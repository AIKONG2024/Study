from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential , load_model
from keras.layers import Dense
from keras.utils import to_categorical  
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping

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
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)


#모델 구현
# model = Sequential()
# model.add(Dense(64, input_dim = 13))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(3, activation='softmax'))

# es = EarlyStopping(monitor='val_loss', mode='min', patience=80, verbose=1, restore_best_weights=True)

# #컴파일 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# history = model.fit(x_train, y_train, epochs=550, batch_size=1, validation_split=0.2, verbose=1,callbacks=[es])

path = '../_data/_save/MCP/wine/'
model = load_model('k26_0117_1215_0104-26.9510.hdf5')

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

# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,6))
# plt.plot(history.history['val_acc'], color = 'blue', label = 'val_acc', marker = '.')
# plt.plot(history.history['val_loss'], color = 'red', label = 'val_loss', marker = '.')
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
'''