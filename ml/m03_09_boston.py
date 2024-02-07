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
r2 = model.score(x_test, y_test)
y_predict = model.predict(x_test)
r2_pred = r2_score(y_test, y_predict)
print('r2 : ', r2)
print('eval_r2 : ', r2_pred)
'''
r2 :  0.6664703504818559
eval_r2 :  0.6664703504818559
'''
