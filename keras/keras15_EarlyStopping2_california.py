
#데이터 수집
from sklearn.datasets import fetch_california_housing

datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

#데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state= 20)

#데이터 구조 확인
print(x_train.shape)#(14447, 8)
print(x_test.shape)#(6193, 8)
print(y_train.shape)#(14447,)
print(y_test.shape)#(6193,)

#모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim = 8))
model.add(Dense(30))
model.add(Dense(1))

#얼리스톱
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode = 'min', patience=15, restore_best_weights=False,  verbose=1) #parience 15일떄 좋음.


# 컴파일, 훈련
model.compile(loss= 'mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=500,batch_size=100,  verbose=1, validation_split= 0.7, callbacks=[es])

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

#restore_best_weights = True
#loss : 0.6844689846038818

#restore_best_weights = False
# loss : 1.311980962753296