# 09_1 카피
import warnings

warnings.filterwarnings("ignore")  # 워닝 무시

from sklearn.datasets import load_boston

datasets = load_boston()
x = datasets.data  # x값
y = datasets.target  # y값

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import time

# 데이터
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.70, random_state=4291451
)


# 모델구성
model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(32))
model.add(Dense(1))

# 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=16, validation_split=0.2)
end_time = time.time()

# 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score, mean_absolute_error

r2 = r2_score(y_test, y_predict)
print("loss : ", loss)
print("R2 : ", r2)


def RMSE(y_test, y_predict):
    return np.sqrt(mean_absolute_error(y_test, y_predict))


rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)
print("걸린시간 : ", round(end_time - start_time, 2), "초")

print("=" * 40 + "hist" + "=" * 40)
print(hist)
print("=" * 40 + "hist.history" + "=" * 40)
print(hist.history)
print("=" * 40 + "loss" + "=" * 40)
print(hist.history["loss"])
print("=" * 40 + "val loss" + "=" * 40)
print(hist.history["val_loss"])
print("=" * 90)

# 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 6))
plt.plot(hist.history["loss"], color="red", label= "loss" , marker=".")
plt.plot(hist.history["val_loss"], color="blue", label="val_loss", marker=".") #color 는 c라고 해도 됨
plt.legend(loc = 'upper right') #lower_left, upper_left, lower_right upper_right
plt.title('boston loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()
