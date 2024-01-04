import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y = np.array([1, 2, 4, 3, 5, 7, 9, 3, 8, 12, 13, 8, 14, 15, 9, 6, 17, 23, 21, 20])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=1643823
)

# 2.모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, epochs=1, batch_size=1)

# 4. 예측, 평가
loss = model.evaluate(x_test, y_test)
result = model.predict(x)

print("로스값: ", loss)
print("예측 값: ", result)
print(f'x_train: {x_train} \ny_train: {y_train} \nx_test : {x_test} \ny_test: {y_test}')

import matplotlib.pyplot as plt

plt.scatter(x,y)
# plt.plot(x, y, color = 'blue')
# plt.scatter(x, result, color = 'red')
plt.plot(x, result, color = 'red')
plt.show()

# 결과기 괜찮게 나온 random_state : 1200
# 결과기 괜찮게 나온 random_state : 1643823