#데이터 가져오기

from sklearn.datasets import load_boston

x = load_boston().data
y = load_boston().target

print(x.shape, y.shape)

from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, random_state= 200)

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='max', patience=20, restore_best_weights=True)

#모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim = 13))
model.add(Dense(64))
model.add(Dense(1))

#훈련 컴파일
model.compile(loss='mse', optimizer= 'adam', metrics=['mse', 'mae', 'acc'])
history = model.fit(x_train, y_train, epochs=100, batch_size= 100, validation_split=0.3, callbacks= [es])

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict =  model.predict(x_test)

print("loss 값 : ", loss)
print("예측 값 : ", y_predict)

history_loss = history.history['loss']
history_val_loss = history.history['val_loss']
history_acc = history.history['acc']
history_val_acc = history.history['val_acc']

print(history_acc)


#시각화
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))
plt.plot(history_loss, color = 'red', label = 'loss', marker = '.')
plt.plot(history_val_loss, color = 'blue', label = 'val_loss', marker = '.')
plt.plot(history_acc, color = 'green', label = 'acc', marker = '.')
plt.plot(history_val_acc, color = 'brown', label = 'val_acc', marker = '.')
plt.show()
