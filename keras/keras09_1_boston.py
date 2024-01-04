import warnings
warnings.filterwarnings('ignore') #워닝 무시

from sklearn.datasets import load_boston
#현재 사이킷런 버전 1.3.0 보스턴 안됨. 
# 그래서 삭제.
# pip uninstall scikit-learn
# pip uninstall scikit-learn-intelex
# pip uninstall scikit-image
# pip install scikit-learn==1.1.3    

datasets = load_boston()
# print(datasets)
x = datasets.data #x값
y = datasets.target #y값
# print(x.shape) #(506,13)
# print(y.shape) #(506,)

# print(datasets.feature_names) #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

# print(datasets.DESCR) #설명

#[실습]
# train_size 0.7이상, 0.9이하
# R2 0.8 이상
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import time

#데이터
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state= 92111113)

#모델구성
model = Sequential()
model.add(Dense(1, input_dim = 13))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
model.fit(x_train, y_train, epochs=300, batch_size=1)
end_time = time.time()

#평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print("loss : ", loss)
print("R2_score :", r2)
print("걸린시간 : ", round(end_time - start_time, 2), "초")


#train_size : 0.7 /  deep 3 (13-64-64-1)  / random_state : 70 / epochs = 100 / batch_size = 1
# loss :  35.30439758300781
# R2_score : 0.6355533207640938

#train_size : 0.7 /  deep 3 (13-64-64-1)  / random_state : 70 / epochs = 300 / batch_size = 1
# loss :  30.12049674987793
# R2_score : 0.6890666151822387

#train_size : 0.7 /  deep 3 (13-64-64-1)  / random_state : 70 / epochs = 300 / batch_size = 1

#train_size : 0.7 /  deep 3 (13-1-64-64-1)  / random_state : 921111 / epochs = 200 / batch_size = 1
# loss :  24.854013442993164
# R2_score : 0.7190490529796778