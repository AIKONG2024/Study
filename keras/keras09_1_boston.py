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
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.70, random_state= 4291451)


#모델구성
model = Sequential()
model.add(Dense(64, input_dim = 13))
model.add(Dense(32))
model.add(Dense(1))

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=16)
end_time = time.time()

#평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print("loss : ", loss)
print("R2 : ", r2)
print("걸린시간 : ", round(end_time - start_time, 2), "초")


#train_size : 0.7 /  deep 5 (13-64-64-64-1)  / random_state : 4291451 / epochs = 4029 / batch_size = 515
# loss :  19.533924102783203
# R2 :  0.7772270803659531
# 걸린시간 :  5.8 초

