import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate, Concatenate 

# 1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T  # 삼성전자 종가, 하이닉스 종가

y1 = np.array(range(3001, 3101))  # 비트코인 종가
y2 = np.array(range(13001, 13101))  # 이더리움 종가

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datasets, y1, y2, train_size=0.7, random_state=123,
)

# print(x1_train.shape, x2_train.shape, y1_train.shape) #(70, 2) (70, 3) (70,)

#2-1 모델
input1 = Input(shape=(2,))
print(input1)
dense1 = Dense(10, activation='relu', name = 'bit1')(input1) #Name 은 레이어의 성능에 영향을 주지 않음. 그냥 레이어 이름 지정
dense2 = Dense(10, activation='relu', name = 'bit2')(dense1)
dense3 = Dense(10, activation='relu', name = 'bit3')(dense2)
output1 = Dense(1, activation='relu', name = 'bit4')(dense3)
output2 = Dense(1, activation='relu', name = 'bit5')(dense3)
model = Model(inputs = input1, outputs = [output1, output2])

model.summary()

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x1_train, [y1_train, y2_train], epochs=1000)

#4. 평가 예측
loss = model.evaluate(x1_test, [y1_test, y2_test])
print("loss :", loss)
predict = model.predict(x1_test)
print("predict :", predict)

'''
loss : [15325.3974609375, 234.66943359375, 15090.728515625]
'''
