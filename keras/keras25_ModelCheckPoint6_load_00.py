# save_best_only
# restore_best_weights
# 에 대한 고찰!!!

import numpy as np

# 데이터 가져오기
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

datasets = load_boston()
x = datasets.data
y = datasets.target

# 데이터 분석
print(x.shape)
print(y.shape)

# 데이터 전처리
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=20
)
# 스케일러는 split 후에 해야 x_train의 기준과 동일하게 x_test의 기준을 정해줌.
# predict 할 값도 train 의 기준에 맞춰야함.
from sklearn.preprocessing import (
    MinMaxScaler,
    MaxAbsScaler,
    StandardScaler,
    RobustScaler,
)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train))  # 0.0
print(np.min(x_test))  # -0.028657616892911006
print(np.max(x_train))  # 1.0000000000000002
print(np.max(x_train))  # 0.0

# 데이터 구조 확인
print(x_train.shape)  # (301, 13)
print(x_test.shape)  # (152, 13)
print(y_train.shape)  # (354,)
print(y_test.shape)  # (152,)

# 모델 구성
from keras.models import Sequential, load_model
from keras.layers import Dense

# model = Sequential()
# model.add(Dense(20, input_dim=13))
# model.add(Dense(10))
# model.add(Dense(1))
path = '../_data/_save/MCP/'
model = load_model(path + 'k25_0117_1215_0104-26.9510.hdf5')
model.summary()

print(model.history)

# 컴파일, 훈련
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# import datetime

# date = datetime.datetime.now()
# print(date)  # 2024-01-17 10:52:41.770061
# date = date.strftime("%m%d_%H%M")
# print(date)


# path = "../_data/_save/MCP/"
# filename = "{epoch:04d}-{val_loss:.4f}.hdf5"
# filepath = "".join([path, "k25_", date, "_", filename])  


# es = EarlyStopping(
#     monitor="val_loss", mode="min", patience=10, verbose=1, restore_best_weights=False
# )
# mcp = ModelCheckpoint(
#     monitor="val_loss", mode="min", verbose=1, save_best_only=True, filepath=filepath
# )
# model.compile(loss="mse", optimizer="adam")
# hist = model.fit(
#     x_train,
#     y_train,
#     epochs=1000,
#     batch_size=10,
#     validation_split=0.7,
#     verbose=1,
#     callbacks=[es, mcp],
# )

# 평가 예측
print("=================1. 기본출력 ====================")
loss = model.evaluate(x_test, y_test, verbose=0)
y_predict = model.predict(x_test, verbose=0)
r2 = r2_score(y_test, y_predict)

print("loss : ", loss)
print("r2 : ", r2_score)

print("==" * 50)
# print(hist.history["val_loss"])
print("==" * 50)


# restore_best_weights = #save_best_only
#                      =
#True,                 = True           #history 체크포인트 가장 좋은 결과 저장
#True,                 = False          #history 얼리스톱 전 모든 값 저장 
#False,                = True           #얼리스톱 안걸린 밀린 값의 모든 값 중 가중치 갱신된 값 저장
#False,                = False          #에포에 대한 모든 값이 저장

