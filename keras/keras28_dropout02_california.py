
#데이터 수집
from sklearn.datasets import fetch_california_housing

datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

#데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state= 20)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#데이터 구조 확인
print(x_train.shape)#(14447, 8)
print(x_test.shape)#(6193, 8)
print(y_train.shape)#(14447,)
print(y_test.shape)#(6193,)

#모델 구성
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(10, input_dim = 8))
model.add(Dense(30))
model.add(Dropout(0.3))
model.add(Dense(1))

#ModelCheckPoint
from keras.callbacks import ModelCheckpoint
import datetime
date = datetime.datetime.now()
print(date) #2024-01-17 10:52:41.770061
date = date.strftime("%m%d_%H%M")
print(date)


mcp_path = '../_data/_save/MCP/california/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([mcp_path, 'k26_02_california_' ,date, '_', filename]) #체크포인트 가장 좋은 결과들 저장
mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only= True, filepath=filepath)


# 컴파일, 훈련
model.compile(loss= 'mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=10, verbose=1, validation_split= 0.7, callbacks=[mcp])

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print("loss :", loss)
print("예측값 : ", y_predict)

#r2 _score
from sklearn.metrics import r2_score
r2_score = r2_score(y_test, y_predict)
print("r2 score : ", r2_score)

#시각화
import matplotlib.pyplot as plt
history_loss = hist.history['loss']
history_val_loss = hist.history['val_loss']

#한글깨짐 처리
plt.rcParams['font.family'] ='Malgun Gothic'

plt.figure(figsize=(6,6)) #세로 9 가로 6
plt.plot(history_loss, c = 'red', label = 'loss', marker = '.' )
plt.plot(history_val_loss, c = 'blue', label = 'val_loss', marker = '.' )
plt.legend(loc = 'upper right')
plt.title('캘리포니아 찻트')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()

'''
기존 : 
loss :  4.23959493637085
============================
best : MaxAbs
============================
MinMaxScaler()
 - loss : 0.5662848949432373
StandardScaler()
 - loss : 0.5383182764053345
MaxAbsScaler()
 - loss : 0.6582180857658386
RobustScaler()
 - loss : 0.6482890248298645
 
============================
Dropout() 적용:
 loss : 1.2975385189056396
'''
