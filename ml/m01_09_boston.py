import numpy as np
from sklearn.datasets import load_boston
datasets= load_boston()
x = datasets.data
y = datasets.target

#데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=20)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#모델 구성
from sklearn.svm import LinearSVR
model = LinearSVR(C=150)
model.fit(x_train, y_train)

#평가 예측
from sklearn.metrics import r2_score
loss = model.score(x_test, y_test)
y_predict = model.predict(x_test)
r2_score = r2_score(y_test, y_predict)

print('r2 : ', loss)
print('r2_score : ', r2_score)

'''
기존 : 
loss : 181.03074645996094
============================
r2 :  0.6616229320178135
r2_score :  0.6616229320178135
'''
