import numpy as np
#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

## 잘라라!!! split
#11,12,13 훈련값
#14,15,16 을 테스트값
x_train = x[10:13]
y_train = y[10:13]
x_test = x[13:16]
y_test = y[13:16]

#데이터 구조 확인
print(x.shape)
print(y.shape)


#모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(10, input_dim = 1))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(1))

#컴파일 ,훈련
model.compile(loss='mse', optimizer='adam')
import time as tm 
start_time = tm.time()
model.fit(x_train, y_train, epochs= 400, batch_size= 10, verbose=1)
end_time = tm.time()

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
r2_score = r2_score(y_test, y_predict)
print("loss: ", loss)
print("r2_score :", r2_score)
print("예측값 : ", y_predict)

#mse
def MSE(y_test, y_predict):
    return mean_squared_error(y_test, y_predict)

#rmse
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

#rmsle
def RMSLE(y_test, y_predict):
    return np.sqrt(mean_squared_log_error(y_test, y_predict))

print("mse loss: ",  MSE(y_test, y_predict)) # model loss 값과 동일.
print("rmse loss: ", RMSE(y_test, y_predict))
print("rmsle loss :", RMSLE(y_test, y_predict))

print("verbose 걸린시간: ", end_time - start_time)

# print(csv[csv['count'] < 0].count()) csv의 음수 개수 구하기

#verbose 1 loss + 프로그래스바 : 2.6035029888153076
#verbose 0 무시 : 1.9179227352142334
#verbose 2 프로그래스바 제거: 2.2349584102630615
#verbose 3 에포만나옴 : 2.061429023742676
#빠른순서 : 0 > 3 > 2 > 1