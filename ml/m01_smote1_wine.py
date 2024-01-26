import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(178, 13) (178,)
print(np.unique(y, return_counts=True))
print(pd.value_counts(y))
print(y)

x = x[:-35]
y = y[:-35]
print(y)

print(np.unique(y, return_counts=True)) #(array([0, 1, 2]), array([59, 71, 18], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.75,shuffle=True, random_state=777,stratify=y)
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# from keras.models import Sequential
# from keras.layers import Dense

# input_layer = 13
# output_layer = 3
# hidden_layer = 16*int(2/3)
# #2.모델
# model = Sequential()
# model.add(Dense(64, input_shape = (13,),activation='relu'))
# model.add(Dense(128))
# model.add(Dense(3, activation='softmax'))

'''
#3.컴파일
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[
    EarlyStopping(monitor="val_loss", mode="min", patience=1000, restore_best_weights=True)
])

#4.평가 예측
loss = model.evaluate(x_test, y_test)
predict = np.argmax(model.predict(x_test), axis=1) 
print('loss :', loss[0])
print('acc :', loss[1])

f1_score = f1_score( y_test, predict, average='macro')
print("f1_score : ", f1_score)

'''
0.9743209876543211
'''
'''

#100에포일때
########################smote##############################
print("==================SMOTE=====================")
from imblearn.over_sampling import SMOTE
import sklearn as sk
print('사이킷런 : ', sk.__version__) # 1.1.3

x_train, y_train = SMOTE(random_state=777).fit_resample(x_train, y_train)
print(pd.value_counts(y_train))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from keras.models import Sequential
from keras.layers import Dense

input_layer = 13
output_layer = 3
hidden_layer = 16*int(2/3)
#2.모델
model = Sequential()
model.add(Dense(64, input_shape = (13,),activation='relu'))
model.add(Dense(128))
model.add(Dense(3, activation='softmax'))

#3.컴파일
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[
    EarlyStopping(monitor="val_loss", mode="min", patience=1000, restore_best_weights=True)
])

#4.평가 예측
loss = model.evaluate(x_test, y_test)
predict = np.argmax(model.predict(x_test), axis=1) 
print('loss :', loss[0])
print('acc :', loss[1])

f1_score = f1_score( y_test, predict, average='macro')
print("f1_score : ", f1_score)

'''
0.9743209876543211
'''