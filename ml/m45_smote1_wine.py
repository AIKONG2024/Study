import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

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

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.75,shuffle=True, random_state=777,stratify=y)
print(np.unique(y_train, return_counts=True))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
model = RandomForestClassifier(random_state=42)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print("최종점수 :" ,result)
x_predict = model.predict(x_test)
acc = accuracy_score(y_test, x_predict)
print("acc_score :", acc)
f1 = f1_score(y_test, x_predict, average='macro')
print("f1_score : ", f1)

from imblearn.over_sampling import SMOTE
x_train, y_train = SMOTE(random_state=42).fit_resample(x_train, y_train)
print("SMOTE 적용 후")
print(np.unique(y_train, return_counts=True)) 

#2. 모델 구성
model = RandomForestClassifier(random_state=42)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print("최종점수 :" ,result)
x_predict = model.predict(x_test)
acc = accuracy_score(y_test, x_predict)
print("acc_score :", acc)
f1 = f1_score(y_test, x_predict, average='macro')
print("f1_score : ", f1)

'''
(array([0, 1, 2]), array([44, 53, 10], dtype=int64))
최종점수 : 1.0
acc_score : 1.0
f1_score :  1.0
SMOTE 적용 후
(array([0, 1, 2]), array([53, 53, 53], dtype=int64))
최종점수 : 1.0
acc_score : 1.0
f1_score :  1.0
'''