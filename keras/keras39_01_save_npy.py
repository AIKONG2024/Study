# x, y 추출해서 모델 만들기
# 성능 0.99 이상
# 변환시간 체크
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # 각각의 이미지 사이즈를 맞춰줌
    horizontal_flip=True,  # 수평 뒤집기
    vertical_flip=True,  # 수직 뒤집기
    width_shift_range=0.1,  # 평행이동
    height_shift_range=0.1,  # 평행이동
    rotation_range=5,  # 정해진 각도만큼 이미지 회전
    zoom_range=1.2,  # 축소 또는 확대
    shear_range=0.7,  # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
    fill_mode="nearest",  # 이동해서 빈 값을 0이 아닌 최종값과 유사한 근사값으로 정해줌.
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

path_train = "c:/_data/image/brain/train/"
path_test = "c:/_data/image/brain/test/"

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(100, 100),
    batch_size=200,
    class_mode="binary",
    color_mode="grayscale",
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(100, 100),
    batch_size=200,
    class_mode="binary",
    color_mode="grayscale",
)


# 1. 데이터
# x_train = xy_train[0][0]
y_train = xy_train[0][1]

x_test = xy_test[0][0]
y_test = xy_test[0][1]


# print(x_train.shape, y_train.shape)  # (160, 100, 100, 3) (160,)
# print(x_test.shape, y_test.shape)  # (120, 100, 100, 3) (120,)

np_path = '../_data/_save_npy/'
np.save(np_path + 'keras39_1_x_train.npy', arr=xy_train[0][0]) #넘파이 형태 저장
np.save(np_path + 'keras39_1_y_train.npy', arr=xy_train[0][1]) #넘파이 형태 저장
np.save(np_path + 'keras39_1_x_test.npy', arr=xy_test[0][0]) #넘파이 형태 저장
np.save(np_path + 'keras39_1_y_test.npy', arr=xy_test[0][1]) #넘파이 형태 저장

# import matplotlib.pyplot as plt
# plt.imshow(x_train[0])
# plt.show()

# #scaling
# x_train = x_train/255.
# x_test = x_test/255.

# 2. 모델 구성
# from keras.models import Sequential
# from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten

# model = Sequential()
# model.add(Conv2D(64, (3, 3), input_shape=(100, 100, 1), activation="relu"))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# model.add(Dropout(0.2))

# model.add(Conv2D(128, (2, 2), activation="relu"))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# model.add(Dropout(0.3))

# # model.add(Conv2D(256, (2,2), activation='relu'))
# # model.add(MaxPooling2D((2,2), strides=(2,2)))
# # model.add(Dropout(0.4))


# # model.add(Conv2D(256, (2,2), activation='relu'))
# # model.add(MaxPooling2D((2,2), strides=(2,2)))
# # model.add(Dropout(0.5))

# model.add(Flatten())
# model.add(Dense(256, activation="relu"))
# # model.add(Dropout(0.5))
# model.add(Dense(1, activation="sigmoid"))

# from keras.callbacks import EarlyStopping, ModelCheckpoint

# import datetime

# date = datetime.datetime.now()
# print(date)  # 2024-01-17 10:52:41.770061
# date = date.strftime("%m%d_%H%M")
# print(date)
# es = EarlyStopping(
#     monitor="val_loss", mode="min", patience=300, restore_best_weights=True
# )
# mcp_path = "../_data/_save/MCP/IDG/brain"
# filename = "{epoch:04d}-{val_loss:.4f}.hdf5"
# filepath = "".join([mcp_path, "k37_brain_", date, "_", filename])  # 체크포인트 가장 좋은 결과들 저장
# mcp = ModelCheckpoint(
#     monitor="val_loss", mode="min", verbose=1, save_best_only=True, filepath=filepath
# )

# import time as tm

# # 3. 컴파일, 훈련
# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
# startTime = tm.time()
# # model.fit_generator(
# #     xy_train
# model.fit(xy_train,
#           #batch_size #fit_generator에서는 에러, fit 에서는 안먹힘. 
#           steps_per_epoch=16, #전체데이터 / batch_size = 160/10 =16
#           #17은 에러, 15는 배치데이터 손실.
#           epochs=10, 
#         #   validation_data=xy_test, 
#         # validation_split=0.2,
#           callbacks=[])
# # UserWarning: `Model.fit_generator` is deprecated and will
# # be removed in a future version. Please use `Model.fit`, which supports generators.
# endTime = tm.time()
# model.save("../_data/_save/MCP/IDG/brain\k37_brain_save_model.h5")

# # 4.평가 예측
# loss = model.evaluate_generator(xy_test)
# predict = np.round(model.predict(x_test))

# print("loss : ", loss[0])
# print("acc : ", loss[1])

# print("time :", np.round(endTime - startTime, 2), "sec")

"""
loss :  0.0004475515161175281
acc :  1.0

loss :  0.007480014581233263
acc :  1.0
"""
