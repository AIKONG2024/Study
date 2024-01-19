from tensorflow.python.keras.models import Sequential
# from tensorflow.keras.models import Sequential
# from keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D

model = Sequential()
# model.add(Dense(10, input_shape = (3,))) #cnn에서 인풋은 (n,3)
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(10,10,1))) #10은 다음 레이어에 전달할 출력수
model.add(Dense(5))
model.add(Dense(1))

