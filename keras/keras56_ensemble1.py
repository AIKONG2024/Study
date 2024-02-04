import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate, Concatenate 

# 1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T  # 삼성전자 종가, 하이닉스 종가
x2_datasets = np.array(
    [range(101, 201), range(411, 511), range(150, 250)]
).T  # 원유 환율 금시세

print(x1_datasets.shape, x2_datasets.shape)#(100, 2) (100, 3)

y = np.array(range(3001, 3101))  # 비트코인 종가

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(
    x1_datasets, x2_datasets, y, train_size=0.7, random_state=123,
)

# x2_train, x2_test, y2_train, y2_test = train_test_split(
#     x2_datasets, y, train_size=0.7, random_state=123,
# )
print(x1_train.shape, x2_train.shape, y1_train.shape) #(70, 2) (70, 3) (70,)

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

#2-3. concatnate
merge1 = concatenate([output1, output11], name='mg1') #concatenate 도 레이어. merge
merge2 = Dense(7, name = "mg2")(merge1)
merge3 = Dense(11, name = "mg3")(merge2)
last_output = Dense(1, name = "last")(merge3)

model = Model(inputs = [input1, input11], outputs = last_output)

model.summary()

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train], y1_train, epochs=1000)

#4. 평가 예측
loss = model.evaluate([x1_test, x2_test], y1_test)
print("loss :", loss)
predict = model.predict([x1_test, x2_test])
print("leng :", len(predict))
print("predict :", predict)

'''
 [[3008.9993]
 [3071.0002]
 [3083.0005]
 [3028.9995]
 [3064.0002]
 [3000.3193]
 [3005.999 ]
 [3051.    ]
 [3082.0005]
 [3004.9995]
 [3023.9993]
 [3066.0005]
 [3077.0005]
 [3061.    ]
 [3024.9995]
 [3043.    ]
 [3078.0005]
 [3039.    ]
 [3057.    ]
 [3076.0005]
 [3035.9995]
 [3089.0007]
 [3019.9995]
 [3029.9998]
 [3031.9995]
 [3092.0007]
 [3086.0002]
 [3009.9995]
 [3053.9998]
 [3041.9998]]
'''