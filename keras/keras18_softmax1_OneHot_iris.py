import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#1. 데이터
datasets = load_iris()
# print(datasets)
# print(datasets.DESCR)
#4갸의 컬럼, 3개의 클래스(라벨)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(150, 4) (150,) #값이 커지면 회귀데이터라고 판단할 수 있음.
print(y)
print(pd.value_counts(y))#데이터의 종류 볼 수 있음
# print(np.unique(y, return_counts=True)) 
'''
0    50
1    50
2    50
입력값이 50,50,50인 이유
'''


# keras 방식
# from keras.utils import to_categorical
# enc_y = to_categorical(y) 
# print(enc_y)
'''
Name: count, dtype: int64
[[1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]s
'''

#pandas 방식
# enc_y = pd.get_dummies(y)
# print(enc_y)
'''
0     True  False  False
1     True  False  False
2     True  False  False
3     True  False  False
'''

#scikit learn 방식
from sklearn.preprocessing import OneHotEncoder
y = y.reshape(-1,1) # 벡터를 행렬로 바꾼거 앞에 -1은 데이터가 몇개인지 몰라서 써준것.
# y = y.reshape(150,1)

# enc = OneHotEncoder(sparse=False).fit(y) #-n의 크기에 맞추어 형태 정함
# enc_y = enc.transform(y)
# print(enc_y) #(150, 3)
ohe = OneHotEncoder(sparse=False) #toarray() 붙여도됨
# ohe.fit(y) #변수에 저장을 함.
# y_ohe3 = ohe.transform(y) #fit 된 데이터를 transform 에서 변환
y_ohe3 = ohe.fit_transform(y) #fit 과 trainsform 둘 다 

print(y_ohe3)
print(y_ohe3.shape)

'''
Name: count, dtype: int64
[[1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
'''

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe3, train_size=0.8, random_state=13, stratify=y) 
#stratify는 분류에서민 사용
print(np.unique(y_test, return_counts=True))


#모델구성
model = Sequential()
model.add(Dense(64, input_dim = 4))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(3, activation= 'softmax')) #클래스의 개수

es = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)

#컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) #분류 모델에서는 metrics 'acc'
model.fit(x_train, y_train, epochs=200, batch_size=30, verbose=1, validation_split=0.2, 
          callbacks=[es])
import keras

#평가, 예측

print(y_test)
results = model.evaluate(x_test, y_test) #metrics 에 loss, acc 가 들어잇음. compile에 acc를 넣어줬기때문!
y_predict = model.predict(x_test)

arg_test = np.argmax(y_test, axis=1)
arg_predict = np.argmax(y_predict, axis=1)

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(arg_test, arg_predict)
print("로스: ", results[0])
print("정확도: ", results[1])

print("accuracy_score : ", acc_score)
print("예측값: ", arg_predict)
#3개중 하나 선택.
# accuracy_score(y_test, y_predict)


# print(sububmission)


