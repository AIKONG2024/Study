import numpy as np
from sklearn.model_selection import train_test_split
#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))
#비율 : 10 : 3 : 3
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.63, random_state=20)
print(x_train.shape)
print(x_test.shape)
print("===========")
x_test, x_val, y_test, y_val = train_test_split(x_test,y_test, train_size=0.5, random_state=20)
print(x_train.shape)
print(x_test.shape)
print(x_val.shape)

print()

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
model.fit(x_train, y_train, epochs= 5000, batch_size= 100, verbose=0, validation_batch_size=(x_val, y_val))
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


'''
epoch : 1000, batch = 100
loss:  0.00011908973101526499
r2_score : 0.9999896941580437
예측값 :  [[11.993675]
 [ 6.012567]
 [13.987377]]
mse loss:  0.0001190897292720668
rmse loss:  0.010912824074091308
rmsle loss : 0.0011779688039022511
verbose 걸린시간:  1.2952773571014404
'''