#09_1에서 가져옴
import numpy as np
#데이터 가져오기
from sklearn.datasets import load_boston
datasets= load_boston()
x = datasets.data
y = datasets.target

#데이터 분석
print(x.shape)
print(y.shape)

#데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=20)
#스케일러는 split 후에 해야 x_train의 기준과 동일하게 x_test의 기준을 정해줌.
#predict 할 값도 train 의 기준에 맞춰야함.
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



print(np.min(x_train)) #0.0
print(np.min(x_test)) #-0.028657616892911006
print(np.max(x_train)) #1.0000000000000002
print(np.max(x_train)) #0.0


'''

'''

#데이터 구조 확인
print(x_train.shape)#(301, 13)
print(x_test.shape)#(152, 13)
print(y_train.shape)#(354,)
print(y_test.shape)#(152,)

#모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(20, input_dim = 13))
model.add(Dense(10))
model.add(Dense(1))

model.summary()
# model.save("c:\_data\_save\keras24_save_model.h5") #절대경로
model.save("..\_data\_save\keras24_save_model.h5") #상대경로

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs= 10, batch_size= 10, validation_split=0.7, verbose=1)



#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print('loss : ', loss)
# print('result : ', y_predict)

'''
기존 : 
loss :  369.1446228027344
============================
best : MaxAbs
============================
MinMaxScaler()
 - loss : 209.1923065185547
StandardScaler()
 - loss : 513.5611572265625
MaxAbsScaler()
 - loss : 181.03074645996094
RobustScaler()
 - loss : 410.6644287109375
'''
