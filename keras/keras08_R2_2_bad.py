
# 고의적으로 R2 값 낮추기.
#1. R2를 음수가 아닌 0.5 미만으로 만들것
# 2. 데이터는 건들지 말것
# 3. 레이어는 인풋과 아웃풋 포함해서 7개 이상
# 4.batch_size = 1
# 5. 히든레이어의 노드는 10개 이상 100개 이하
# 6. train 사이즈 85%
# 7. epoch 100번 이상
#  [실습 시작]

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y = np.array([1, 2, 4, 3, 5, 7, 9, 3, 8, 12, 13, 8, 14, 15, 9, 6, 17, 23, 21, 20])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.85, random_state=22
)

# 2.모델 구성
model = Sequential()
model.add(Dense(23, input_dim=1))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(33))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4. 예측, 평가
loss = model.evaluate(x_test, y_test) #y_test값과 x_test를 모델에 넣어서 나온 출력값을 비교한 것.
print("로스값: ", loss)

y_predict = model.predict(x_test) 
result = model.predict(x)


#평가지표
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict) #y의 테스트용 실제값과 x를 모델에 돌려 나온 예측값 y_predict를 비교함
print("R2 스코어 : ",r2)

import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.plot(x, result, color = 'red')
plt.show()

# train_size = 0.85, ramdom_state : 22, deep : 8 (1-23-5-10-15-20-33-1) epochs = 100, batch_size =1
# 로스값:  22.7119083404541
# R2 스코어 :  0.3448488150141884



#Good 모델
# 로스값:  7.313045978546143
# R2 스코어 :  0.795756669394788

# train_size = 0.7, ramdom_state : 2600, deep : 3 (1-64-64-1) epochs = 1000, batch_size =1
# R2 스코어 :  0.9514988408834697

# train_size = 0.7, ramdom_state : 2600, deep : 3 (1-64-64-1) epochs = 1000, batch_size =20
# 로스값:  2.062253952026367
# R2 스코어 :  0.94145019982536

# train_size = 0.7, ramdom_state : 92111113, deep : 3 (1-64-64-1) epochs = 1000, batch_size =20
# 로스값:  1.2238337993621826
# R2 스코어 :  0.9615213827865631



# 1. 데이터
# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
# y = np.array([1, 2, 4, 3, 5, 7, 9, 3, 8, 12, 13, 8, 14, 15, 9, 6, 17, 23, 21, 20])

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.85, random_state=55
# )

# # 2.모델 구성
# model = Sequential()
# model.add(Dense(3, input_dim=1))
# model.add(Dense(15))
# model.add(Dense(3))
# model.add(Dense(5))
# model.add(Dense(22))
# model.add(Dense(1))
# model.add(Dense(17))
# model.add(Dense(1))


# # 3. 컴파일, 훈련
# model.compile(loss="mse", optimizer="adam")
# model.fit(x_train, y_train, epochs=100, batch_size=1)

# # 4. 예측, 평가
# loss = model.evaluate(x_test, y_test) #y_test값과 x_test를 모델에 넣어서 나온 출력값을 비교한 것.
# print("로스값: ", loss)

# y_predict = model.predict(x_test) 
# result = model.predict(x)


# #평가지표
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test,y_predict) #y의 테스트용 실제값과 x를 모델에 돌려 나온 예측값 y_predict를 비교함
# print("R2 스코어 : ",r2)

# import matplotlib.pyplot as plt
# plt.scatter(x,y)
# plt.plot(x, result, color = 'red')
# plt.show()

# R2 스코어 :  0.9956075982117933