import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate, Concatenate 
from sklearn.metrics import r2_score

# 1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T  # 삼성전자 종가, 하이닉스 종가
x2_datasets = np.array(
    [range(101, 201), range(411, 511), range(150, 250)]
).T  # 원유 환율 금시세
x3_datasets = np.array([range(100), range(301,401),
                        range(77,177), range(33,133)]).T 

print(x1_datasets.shape, x2_datasets.shape)#(100, 2) (100, 3)

y1 = np.array(range(3001, 3101))  # 비트코인 종가
y2 = np.array(range(13001, 13101))  # 이더리움 종가

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets ,y1, y2, train_size=0.7, random_state=123,
)

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

input111 = Input(shape=(4,))
dense111 = Dense(32, activation='relu', name = 'bit15')(input111)
dense121 = Dense(64, activation='relu', name = 'bit16')(dense111)
dense131 = Dense(64, activation='relu', name = 'bit17')(dense121)
output111 = Dense(3, activation='relu', name = 'bit18')(dense131)

#2-3. concatnate
concatenate_class  = Concatenate()
# merge1 = concatenate([output1, output11, output111], name='mg1') #concatenate 도 레이어. 
merge1 = concatenate_class([output1, output11, output111])
merge2 = Dense(7, name = "mg2")(merge1)
merge3 = Dense(11, name = "mg3")(merge2)
last_output1 = Dense(1, name = "last1")(merge3)

# merge4 = concatenate([output1, output11, output111], name='mg6') #concatenate 도 레이어. merge
merge4 = concatenate_class([output1, output11, output111]) #concatenate 도 레이어. merge
merge5 = Dense(24, name = "mg4")(merge4)
merge6 = Dense(43, name = "mg5")(merge5)
last_output2 = Dense(1, name = "last2")(merge6)

model = Model(inputs = [input1, input11, input111], outputs = [last_output1, last_output2] )

model.summary()

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train, x3_train], [y1_train,y2_train], epochs=1000)

#4. 평가 예측
loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
print("loss :", loss)
predict = model.predict([x1_test, x2_test, x3_test])
print("predict :", predict)
print(predict[0])
print(predict[1])
r2_score1 = r2_score(y1_test, predict[0])
r2_score2 = r2_score(y2_test, predict[1])
# r2_total = r2_score([y1_test, y2_test], [predict[0], predict[1]])
# print(r2_total)
print("y1_r2 :" , r2_score1,"y2_r2 :", r2_score2)


'''
y1_r2 : 0.9999999995902381 y2_r2 : 0.9999999992214524
loss : [18.587120056152344, 1.0757142305374146, 17.51140594482422] #y1 loss, y2 loss, y1+y2의 loss
'''