# x, y 추출해서 모델 만들기
# 성능 0.99 이상
# 변환시간 체크
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


np_path = '../_data/_save_npy/'
# np.save(np_path + 'keras39_1_x_train.npy', arr=xy_train[0][0]) #넘파이 형태 저장
# np.save(np_path + 'keras39_1_y_train.npy', arr=xy_train[0][1]) #넘파이 형태 저장
# np.save(np_path + 'keras39_1_x_test.npy', arr=xy_test[0][0]) #넘파이 형태 저장
# np.save(np_path + 'keras39_1_y_test.npy', arr=xy_train[0][1]) #넘파이 형태 저장

x_train = np.load(np_path + 'keras39_1_x_train.npy')
y_train = np.load(np_path + 'keras39_1_y_train.npy')
x_test = np.load(np_path + 'keras39_1_x_test.npy')
y_test = np.load(np_path + 'keras39_1_y_test.npy')

print(x_train.shape, y_train.shape)#(160, 100, 100, 1) (160,)
print(x_test.shape, y_test.shape)#(120, 100, 100, 1) (160,)

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(100, 100, 1), activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (2, 2), activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Dropout(0.3))

# model.add(Conv2D(256, (2,2), activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))
# model.add(Dropout(0.4))


# model.add(Conv2D(256, (2,2), activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))
# model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
# model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

from keras.callbacks import EarlyStopping, ModelCheckpoint

import datetime

date = datetime.datetime.now()
print(date)  # 2024-01-17 10:52:41.770061
date = date.strftime("%m%d_%H%M")
print(date)
es = EarlyStopping(
    monitor="val_loss", mode="min", patience=300, restore_best_weights=True
)
mcp_path = "../_data/_save/MCP/IDG/brain"
filename = "{epoch:04d}-{val_loss:.4f}.hdf5"
filepath = "".join([mcp_path, "k37_brain_", date, "_", filename])  # 체크포인트 가장 좋은 결과들 저장
mcp = ModelCheckpoint(
    monitor="val_loss", mode="min", verbose=1, save_best_only=True, filepath=filepath
)

import time as tm

# 3. 컴파일, 훈련
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
startTime = tm.time()
# model.fit_generator(
#     xy_train
model.fit(x_train, y_train,
          #batch_size #fit_generator에서는 에러, fit 에서는 안먹힘. 
          steps_per_epoch=16, #전체데이터 / batch_size = 160/10 =16
          #17은 에러, 15는 배치데이터 손실.
          epochs=10, 
        #   validation_data=xy_test, 
          validation_split=0.2,
          callbacks=[])
# UserWarning: `Model.fit_generator` is deprecated and will
# be removed in a future version. Please use `Model.fit`, which supports generators.
endTime = tm.time()
model.save("../_data/_save/MCP/IDG/brain\k37_brain_save_model.h5")
loss = model.evaluate(x_test, y_test)
# 4.평가 예측
predict = np.round(model.predict(x_test))

print("loss : ", loss[0])
print("acc : ", loss[1])

print("time :", np.round(endTime - startTime, 2), "sec")

"""
loss :  0.0004475515161175281
acc :  1.0

loss :  0.007480014581233263
acc :  1.0
"""
