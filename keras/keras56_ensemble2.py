import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate, Concatenate 

# 1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T  # 삼성전자 종가, 하이닉스 종가
x2_datasets = np.array(
    [range(101, 201), range(411, 511), range(150, 250)]
).T  # 원유 환율 금시세
x3_datasets = np.array([range(100), range(301,401),
                        range(77,177), range(33,133)]).T

print(x1_datasets.shape, x2_datasets.shape)#(100, 2) (100, 3)

y = np.array(range(3001, 3101))  # 비트코인 종가

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets ,y, train_size=0.7, random_state=123,
)

print(x1_train.shape, x2_train.shape, x3_train, y1_train.shape) #(70, 2) (70, 3) (70,)

#2-1 모델
input1 = Input(shape=(2,))
print(input1)
dense1 = Dense(10, activation='relu', name = 'bit1')(input1) #Name 은 레이어의 성능에 영향을 주지 않음. 그냥 레이어 이름 지정
dense2 = Dense(10, activation='relu', name = 'bit2')(dense1)
dense3 = Dense(10, activation='relu', name = 'bit3')(dense2)
output1 = Dense(10, activation='relu', name = 'bit4')(dense3)
# model1 = Model(inputs = input1, outputs = output1)
# model1.summary(0)

#2-2 모델
input11 = Input(shape=(3,))
dense11 = Dense(100, activation='relu', name = 'bit5')(input11)
dense12 = Dense(100, activation='relu', name = 'bit6')(dense11)
dense13 = Dense(100, activation='relu', name = 'bit7')(dense12)
output11 = Dense(5, activation='relu', name = 'bit8')(dense13)
# model2 = Model(inputs = input11, outputs = output11)
# model2.summary(0)

input111 = Input(shape=(4,))
dense111 = Dense(32, activation='relu', name = 'bit15')(input111)
dense121 = Dense(32, activation='relu', name = 'bit16')(dense111)
dense131 = Dense(32, activation='relu', name = 'bit17')(dense121)
output111 = Dense(3, activation='relu', name = 'bit18')(dense131)

#2-3. concatnate
merge1 = concatenate([output1, output11, output111], name='mg1') #concatenate 도 레이어. merge
merge2 = Dense(7, name = "mg2")(merge1)
merge3 = Dense(11, name = "mg3")(merge2)
last_output = Dense(1, name = "last")(merge3)

model = Model(inputs = [input1, input11, input111], outputs = last_output)

model.summary()

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train, x3_train], y1_train, epochs=1000)

#4. 평가 예측
loss = model.evaluate([x1_test, x2_test, x3_test], y1_test)
print("loss :", loss)
predict = model.predict([x1_test, x2_test, x3_test])
print("predict :", predict)

'''
loss : 0.15042825043201447
'''
