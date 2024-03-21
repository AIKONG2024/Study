


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping

datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

# print(x.shape)
# print(y.shape)

x_train , x_test, y_train, y_test = train_test_split(x,y, train_size= 0.7, random_state= 200)

#모델 구성
model = Sequential()
model.add(Dense(64, input_dim = 8))
model.add(Dense(32))
model.add(Dense(1))

es = EarlyStopping(monitor='val_acc', mode='max', patience= 100, restore_best_weights= True)

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'acc'])
history = model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.3, verbose=1, callbacks=[es])

#평가 예측
loss = model.evaluate(x_test, y_test)
submission = model.predict(x_test)

print(submission)

history_loss = history.history['loss']
history_val_loss = history.history['val_loss']
history_acc = history.history['acc']
history_val_acc = history.history['val_acc']

#시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(history_loss, color = 'red', label = 'loss', marker = '.')
plt.plot(history_val_loss, color = 'blue', label = 'val_loss', marker = '.')
plt.plot(history_acc, color = 'green', label = 'acc', marker = '.')
plt.plot(history_val_acc, color = 'brown', label = 'val_acc', marker = '.')
plt.show()