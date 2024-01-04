import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 6, 5, 7, 8, 9, 10])

#데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size= 0.7,
    test_size=0.3,
    shuffle=True,
    random_state=4294967295
)
'''
test_size 기본값 0.25 , random_state의 범위 : 0 ~ 4294967295
train_size 나 test_size중 하나만 써줌. 왜냐면 train_size : 0.7, test_size : 0.4 라고 하면 합이 1.0을 벗어나기 때문에 에러.
만약 합이 1.0이 안된다면 데이터 손실이 발생함.
1,10을 포함하면 데이터 범위가 증가해서 훈련 결과가 더 좋아짐. 
1,10을 반드시 포함하고 싶다면 random_state 를 직접 설정을 바꿔주면서 최적의 훈련값을 찾아야함. 결과치를 보고 판단.

디폴트값
def train_test_split(
    *arrays,  
    test_size=None, 디폴트값 0.75
    train_size=None,
    random_state=None,
    shuffle=True,
    stratify=None,
):
'''


print(x_train)
print(y_train)
print(x_test)
print(y_test)

#모델구성
model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(5))
model.add(Dense(1))

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=2000 , batch_size=1)

#평가 , 예측
loss = model.evaluate(x_test, y_test)
result = model.predict([200])

print("loss 값: ", loss)
print("예측 값: ", result)
