from sklearn.metrics import r2_score
import time
from keras.layers import Dense, Conv2D, Flatten, LSTM
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import warnings
warnings.filterwarnings('ignore')


# 1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x)
print(y)
print(x.shape, y.shape)  # (442, 10) (442,)

print(dataset.feature_names)
print(dataset.DESCR)
# R2 0.62 이상.
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.72, random_state=335688)  # 335688

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 모델 구성

model = Sequential()
model.add(LSTM(16, input_shape = (10,1)))
model.add(Flatten())
model.add(Dense(64, input_dim=10))
model.add(Dense(32))
model.add(Dense(1))

#Early Stopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=15, verbose=1, restore_best_weights=True)

# 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'acc'])
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=20, validation_split=0.7, callbacks=[es])
end_time = time.time()

# 평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("로스 : ", loss)
print("r2 = ", r2)
print("걸린시간 : ", round(end_time - start_time, 2), "초")

#로스값, val_loss
hist_loss = hist.history['loss']
hist_val_loss = hist.history['val_loss']

#시각화
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.figure(figsize=(6,6))
plt.plot(hist_loss, c = 'red', label = 'loss', marker = '.')
plt.plot(hist_val_loss, c = 'blue', label = 'val_loss', marker = '.')
plt.legend(loc = 'upper right')
plt.xlabel = 'epoch'
plt.ylabel = 'loss'
plt.title = '디아벳 차트'
plt.grid()
plt.show()

#restore_best_weights = True


'''
기존 : 
loss :  [2657.56591796875, 2657.56591796875, 42.625911712646484, 0.0]
============================
best : MaxAbs
============================
MinMaxScaler()
 - loss : [2654.833251953125, 2654.833251953125, 42.52450180053711, 0.0]
StandardScaler()
 - loss :  [2636.103759765625, 2636.103759765625, 41.95334243774414, 0.0]
MaxAbsScaler()
 - loss : [2613.149658203125, 2613.149658203125, 41.775360107421875, 0.0]
RobustScaler()
 - loss : [2850.427490234375, 2850.427490234375, 44.245235443115234, 0.0]
 
================================
CNN 적용
 
 로스 :  [2686.28369140625, 2686.28369140625, 42.94321060180664, 0.0]
r2 =  0.5941362035719142
걸린시간 :  7.29 초

================================
RNN 적용후
로스 :  [4427.11474609375, 4427.11474609375, 56.2730712890625, 0.0]
r2 =  0.3311183963312865
걸린시간 :  6.35 초
'''