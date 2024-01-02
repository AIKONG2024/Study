import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터(잘못 준 데이터) -> 행열 원위치
x = np.array(
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3] # 짜증나는 데이터지만, 로스가 떨어지는 것을 확인하면서 가중치를 구할 수 있음.
    ]  #컬럼 데이터들을 대괄호로 묶어 줘야함.
)
print("numpy 버전: ", np.__version__)

y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print(x.shape)  # (2,10)
print(y.shape)  # (10,)
# x = x.T #전치. 행과 열을 바꿔줌.
# x = x.transpose()
x = x.swapaxes(0,1) 

# [1,1], [2, 1.1], [3, 1.2]... [10, 1.3]
print(x.shape)  # (10, 2)
print(x)
# 2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim = 2)) #input 데이터는 2차원. 
#열 == 컬럼 == 속성 == 특성 == 차원 = 2
#(행 무시 열 우선) <=외우기
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=3000, batch_size= 1)
#4.평가, 예측
loss = model.evaluate(x,y)
results = model.predict([[10, 1.3]]) #!!!!입력과 같은 열을 맞춰줘야함 x.shape = (1,2) ==> y도 (N,2) 형태로 들어가야 함. (행무시 열우선) => (None,2)

# [실습] : 소수점 2째 자리까지 맞추기
print("[10, 1.3]의 예측값 : ", results) 

# deep = 3(2-64-128-64-1), batch_size = 1, epochs = 1000
# [10, 1.3]의 예측값 :  [[9.999687]]

# deep = 3(2-64-128-64-1), batch_size = 1, epochs = 2000
# [10, 1.3]의 예측값 :  [[9.992325]]

# deep = 3(2-64-128-64-1), batch_size = 1, epochs = 3000
# [10, 1.3]의 예측값 :  [[10.003976]]
# [10, 1.3]의 예측값 :  [[9.999341]]
# [10, 1.3]의 예측값 :  [[10.000504]]

#10.01, 9.99
#과제>tensor 이상으로 20문제 만들고 푸는 A4한장.
